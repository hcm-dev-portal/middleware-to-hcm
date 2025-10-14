# backend/app/vector_bootstrap.py
from __future__ import annotations

import os
import time
import asyncio
import logging
import random
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---- Global flags / timeouts (env-tunable) ----
DEFAULT_STARTUP_WARM = os.getenv("VECTOR_STARTUP_WARM", "true").lower() == "true"
DEFAULT_STARTUP_TIMEOUT = int(os.getenv("VECTOR_STARTUP_TIMEOUT_MS", "60000"))  # overall warmup deadline
DEFAULT_SERVICE_TIMEOUT = int(os.getenv("VECTOR_SERVICE_TIMEOUT_MS", "30000"))  # per-service op timeout
DEFAULT_RETRY_BACKOFF_MS = int(os.getenv("VECTOR_STARTUP_RETRY_BACKOFF_MS", "800"))
DEFAULT_MAX_RETRIES = int(os.getenv("VECTOR_STARTUP_MAX_RETRIES", "3"))
BOOTSTRAP_DISABLED = os.getenv("VECTOR_BOOTSTRAP_DISABLED", "0").lower() in ("1", "true")

def _has(obj: Any, name: str) -> bool:
    try:
        attr = getattr(obj, name, None)
        # We consider both callables and simple attrs as "has"
        return attr is not None
    except Exception:
        return False

async def _maybe_async_call(fn_or_attr, *args, **kwargs):
    """
    Call sync/async function uniformly; if a plain attribute is passed, just return it.
    """
    try:
        # If it's a function or bound method, possibly coroutine
        if callable(fn_or_attr):
            if asyncio.iscoroutinefunction(fn_or_attr):
                return await fn_or_attr(*args, **kwargs)
            res = fn_or_attr(*args, **kwargs)
            if asyncio.iscoroutine(res):
                return await res
            return res
        # Plain attribute value
        return fn_or_attr
    except Exception as e:
        raise e

async def _with_timeout(coro_or_func, timeout_ms: int, *args, **kwargs):
    """
    Run a coroutine or sync function with asyncio timeout.
    Accepts either a coroutine/function or a value.
    """
    try:
        if callable(coro_or_func):
            coro = _maybe_async_call(coro_or_func, *args, **kwargs)
        else:
            # Might already be a coroutine/value
            coro = _maybe_async_call(coro_or_func)
        return await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)
    except asyncio.TimeoutError:
        raise TimeoutError(f"operation timed out after {timeout_ms} ms")

class VectorBootstrapper:
    """
    Proactively builds/loads vector indices at startup (and on demand).
    Tolerates different vector service implementations by probing for common methods.
    """
    def __init__(
        self,
        nlp_services: Dict[str, Any],
        *,
        warm_on_start: bool = DEFAULT_STARTUP_WARM,
        startup_timeout_ms: int = DEFAULT_STARTUP_TIMEOUT,
        retry_backoff_ms: int = DEFAULT_RETRY_BACKOFF_MS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        service_timeout_ms: int = DEFAULT_SERVICE_TIMEOUT,
    ) -> None:
        self._nlp_services = nlp_services  # {"primary": obj, "language_native": obj, ...}
        self._warm_on_start = warm_on_start and not BOOTSTRAP_DISABLED
        self._timeout_ms = startup_timeout_ms
        self._service_timeout_ms = service_timeout_ms
        self._retry_backoff_ms = retry_backoff_ms
        self._max_retries = max_retries

        self._started = False
        self._finished = False
        self._error: Optional[str] = None
        self._t0 = 0.0
        self._t1 = 0.0

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "started": self._started,
            "finished": self._finished,
            "duration_ms": int((self._t1 - self._t0) * 1000) if self._finished else None,
            "error": self._error,
            "warm_on_start": self._warm_on_start,
            "timeout_ms": self._timeout_ms,
            "service_timeout_ms": self._service_timeout_ms,
            "max_retries": self._max_retries,
            "disabled": BOOTSTRAP_DISABLED,
        }

    async def _ensure_one_ready(self, name: str, svc: Any) -> Dict[str, Any]:
        """
        Make a single vector service ready. We detect & call:
          1) embeddings readiness: ensure_embeddings_ready() / lazy_load_embeddings()
          2) index build/load: ensure_index_loaded() / build_or_load_index() / load()
          3) warmup: warmup() / prewarm() / warm()
          4) readiness check: vector_status()/health_check()/is_ready()/ready or structural hints
        Each step has a per-service timeout.
        """
        info = {"service": name, "attempts": 0, "ready": False, "error": None, "ms": 0}
        t0 = time.perf_counter()

        async def _is_ready_async(s: Any) -> bool:
            # 1) vector_status()
            if _has(s, "vector_status"):
                try:
                    st = await _with_timeout(getattr(s, "vector_status"), self._service_timeout_ms)
                    if isinstance(st, dict):
                        if "ready" in st:
                            return bool(st["ready"])
                        # some services may return {"status":"ready"} etc.
                        val = st.get("status") or st.get("state")
                        if isinstance(val, str) and val.lower() in ("ready", "ok", "healthy"):
                            return True
                    elif isinstance(st, bool):
                        return st
                except Exception as e:
                    logger.debug("vector_status probe failed for %s: %s", name, e)

            # 2) health_check()
            if _has(s, "health_check"):
                try:
                    hc = await _with_timeout(getattr(s, "health_check"), self._service_timeout_ms)
                    if isinstance(hc, dict):
                        if isinstance(hc.get("ready"), bool):
                            return bool(hc["ready"])
                        # Heuristic: embeddings present + vector_items exist
                        emb_ok = bool(hc.get("embeddings_en_shape") or hc.get("embeddings_zh_shape"))
                        items_ok = (hc.get("vector_items") or 0) > 0
                        if emb_ok and items_ok:
                            return True
                except Exception as e:
                    logger.debug("health_check probe failed for %s: %s", name, e)

            # 3) is_ready() method or property
            if _has(s, "is_ready"):
                try:
                    val = await _with_timeout(getattr(s, "is_ready"), self._service_timeout_ms)
                    if isinstance(val, bool):
                        return val
                except Exception as e:
                    logger.debug("is_ready() probe failed for %s: %s", name, e)
            # property 'ready'
            try:
                if _has(s, "ready"):
                    val = getattr(s, "ready")
                    if isinstance(val, bool):
                        return val
            except Exception:
                pass

            # 4) structural hints (works for LeaveVectorDB)
            try:
                # any index object?
                if getattr(s, "index", None) or getattr(s, "_index", None):
                    return True
                # bilingual indexes/embeddings
                idx_en = getattr(s, "index_en", None)
                idx_zh = getattr(s, "index_zh", None)
                emb_en = getattr(s, "embeddings_en", None)
                emb_zh = getattr(s, "embeddings_zh", None)
                items = getattr(s, "_vector_items", None)
                if (idx_en or idx_zh) and (emb_en is not None or emb_zh is not None) and items:
                    return True
            except Exception:
                pass

            return False

        async def _call_safe(label: str, func_name: str):
            if not _has(svc, func_name):
                return False
            try:
                await _with_timeout(getattr(svc, func_name), self._service_timeout_ms)
                logger.info("vector %s.%s OK", name, label)
                return True
            except Exception as e:
                logger.warning("vector %s.%s failed: %s", name, label, e)
                return False

        # Already ready? (fast path)
        try:
            if await _is_ready_async(svc):
                info["ready"] = True
                info["ms"] = int((time.perf_counter() - t0) * 1000)
                return info
        except Exception:
            # continue normal flow
            pass

        # Else, build & warm with retries
        retries = 0
        while retries <= self._max_retries:
            info["attempts"] += 1
            try:
                # (1) Embeddings readiness (optional)
                await _call_safe("embeddings", "ensure_embeddings_ready") or \
                await _call_safe("embeddings", "lazy_load_embeddings")

                # (2) Build or load an index
                await _call_safe("build_or_load_index", "ensure_index_loaded") or \
                await _call_safe("build_or_load_index", "build_or_load_index") or \
                await _call_safe("build_or_load_index", "load")

                # (3) Warmup / prewarm (optional)
                await _call_safe("warmup", "warmup") or \
                await _call_safe("warmup", "prewarm") or \
                await _call_safe("warmup", "warm")

                # (4) Soft search probe — treat success as ready
                if _has(svc, "search"):
                    try:
                        await _with_timeout(lambda: svc.search("healthcheck", top_k=1), self._service_timeout_ms)
                        # If search doesn't throw, that's a strong signal it's ready
                        info["ready"] = True
                    except Exception as e:
                        logger.debug("vector %s.search probe failed (non-fatal): %s", name, e)

                # Final readiness check
                if info["ready"] or await _is_ready_async(svc):
                    info["ready"] = True
                    break

                raise RuntimeError("vector service not ready after warmup")

            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"
                if retries >= self._max_retries:
                    break
                # jittered backoff
                delay = self._retry_backoff_ms / 1000.0
                delay *= (1.0 + random.random() * 0.25)
                await asyncio.sleep(delay)
                retries += 1

        info["ms"] = int((time.perf_counter() - t0) * 1000)
        return info

    async def start(self) -> Dict[str, Any]:
        """Start bootstrapping all configured services concurrently."""
        if self._started:
            return {"already_started": True, **self.status}
        self._started = True
        self._error = None
        self._finished = False
        self._t0 = time.perf_counter()

        # Honor disable/env/flag quickly
        if BOOTSTRAP_DISABLED:
            self._finished = True
            self._t1 = time.perf_counter()
            logger.warning("Vector bootstrap disabled by env (VECTOR_BOOTSTRAP_DISABLED=1).")
            return {"disabled": True, **self.status}

        if not self._warm_on_start:
            self._finished = True
            self._t1 = time.perf_counter()
            return {"skipped": True, **self.status}

        try:
            tasks = []
            for name, svc in (self._nlp_services or {}).items():
                if not svc:
                    continue
                tasks.append(self._ensure_one_ready(name, svc))

            # If there are no services, finish successfully
            if not tasks:
                self._t1 = time.perf_counter()
                self._finished = True
                return {"ready": True, "summary": {}, **self.status}

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=False),
                timeout=self._timeout_ms / 1000.0
            )
            self._t1 = time.perf_counter()
            self._finished = True

            # summarize readiness
            summary = {
                r["service"]: {"ready": r["ready"], "attempts": r["attempts"], "ms": r["ms"], "error": r["error"]}
                for r in results
            }
            all_ready = all(r["ready"] for r in results) if results else True
            if not all_ready:
                self._error = "One or more vector services failed to warm up"

            logger.info("vector bootstrap finished ready=%s summary=%s", all_ready, summary)
            return {"ready": all_ready, "summary": summary, **self.status}

        except asyncio.TimeoutError:
            self._t1 = time.perf_counter()
            self._finished = True
            self._error = "Startup warmup timed out"
            logger.warning("vector bootstrap timeout after %dms", self._timeout_ms)
            return {"ready": False, "timeout": True, **self.status}
        except Exception as e:
            self._t1 = time.perf_counter()
            self._finished = True
            self._error = f"{type(e).__name__}: {e}"
            logger.exception("vector bootstrap failed: %s", e)
            return {"ready": False, "error": self._error, **self.status}
