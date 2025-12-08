# backend/app/vector_bootstrap.py
from __future__ import annotations

import os
import time
import asyncio
import logging
import random
import json
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    import numpy as np  # optional; we only use if available
except ImportError:
    np = None

# ---- Global flags / timeouts (env-tunable) ----
DEFAULT_STARTUP_WARM = os.getenv("VECTOR_STARTUP_WARM", "true").lower() == "true"
DEFAULT_STARTUP_TIMEOUT = int(os.getenv("VECTOR_STARTUP_TIMEOUT_MS", "60000"))
DEFAULT_SERVICE_TIMEOUT = int(os.getenv("VECTOR_SERVICE_TIMEOUT_MS", "30000"))
DEFAULT_RETRY_BACKOFF_MS = int(os.getenv("VECTOR_STARTUP_RETRY_BACKOFF_MS", "800"))
DEFAULT_MAX_RETRIES = int(os.getenv("VECTOR_STARTUP_MAX_RETRIES", "3"))
BOOTSTRAP_DISABLED = os.getenv("VECTOR_BOOTSTRAP_DISABLED", "0").lower() in ("1", "true")

# NEW: debug knobs
DEFAULT_DEBUG_DIR = os.getenv("VECTOR_DEBUG_DIR", "").strip() or None
DEBUG_DUMP_NUMPY = os.getenv("VECTOR_DEBUG_DUMP_NUMPY", "0").lower() in ("1", "true")


def _has(obj: Any, name: str) -> bool:
    try:
        attr = getattr(obj, name, None)
        return callable(attr)
    except Exception:
        return False




async def _maybe_async_call(fn: Callable, *args, **kwargs):
    """Call sync or async function uniformly in asyncio context."""
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

async def _with_timeout(coro_or_func, timeout_ms: int, *args, **kwargs):
    """
    Run a coroutine or sync function with asyncio timeout.
    Accepts either a coroutine/function reference.
    """
    try:
        if callable(coro_or_func):
            coro = _maybe_async_call(coro_or_func, *args, **kwargs)
        else:
            coro = coro_or_func
        return await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)
    except asyncio.TimeoutError:
        raise TimeoutError(f"operation timed out after {timeout_ms} ms")

class VectorBootstrapper:
    def __init__(
        self,
        nlp_services: Dict[str, Any],
        *,
        warm_on_start: bool = DEFAULT_STARTUP_WARM,
        startup_timeout_ms: int = DEFAULT_STARTUP_TIMEOUT,
        retry_backoff_ms: int = DEFAULT_RETRY_BACKOFF_MS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        service_timeout_ms: int = DEFAULT_SERVICE_TIMEOUT,
        debug_dir: Optional[str] = DEFAULT_DEBUG_DIR,
    ) -> None:
        self._nlp_services = nlp_services
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

        # NEW: store last summary per-service
        self._last_summary: Dict[str, Any] = {}

        # NEW: debug dir for status + npy dumps
        self._debug_dir = debug_dir
        if self._debug_dir:
            try:
                os.makedirs(self._debug_dir, exist_ok=True)
                logger.info("Vector debug_dir=%s", self._debug_dir)
            except Exception as e:
                logger.warning("Failed to create vector debug_dir=%s: %s", self._debug_dir, e)
                self._debug_dir = None

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
            # NEW: expose last summary + debug_dir
            "summary": self._last_summary,
            "debug_dir": self._debug_dir,
        }

    async def _ensure_one_ready(self, name: str, svc: Any) -> Dict[str, Any]:
        info = {
            "service": name,
            "attempts": 0,
            "ready": False,
            "error": None,
            "ms": 0,
            # NEW: richer visibility
            "pre_status": None,
            "post_status": None,
        }
        t0 = time.perf_counter()

        def _get_status(s: Any) -> Dict[str, Any]:
            if _has(s, "vector_status"):
                try:
                    st = s.vector_status() or {}
                    if isinstance(st, dict):
                        return st
                except Exception as e:
                    logger.debug("vector_status probe failed for %s: %s", name, e)
            return {}

        def _is_ready(s: Any) -> bool:
            st = _get_status(s)
            if st:
                ready = bool(st.get("ready"))
                if ready:
                    return True
            try:
                return bool(getattr(s, "index", None) or getattr(s, "_index", None))
            except Exception:
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

        # capture initial status snapshot
        info["pre_status"] = _get_status(svc)

        # already ready? then just dump debug info and exit
        if _is_ready(svc):
            info["ready"] = True
            info["ms"] = int((time.perf_counter() - t0) * 1000)
            info["post_status"] = _get_status(svc)
            self._dump_debug_artifacts(name, svc, info["post_status"])
            return info

        retries = 0
        while retries <= self._max_retries:
            info["attempts"] += 1
            try:
                await _call_safe("embeddings", "ensure_embeddings_ready") or \
                await _call_safe("embeddings", "lazy_load_embeddings")

                built = await _call_safe("build_or_load_index", "ensure_index_loaded") or \
                        await _call_safe("build_or_load_index", "build_or_load_index") or \
                        await _call_safe("build_or_load_index", "load")

                warmed = await _call_safe("warmup", "warmup") or \
                         await _call_safe("warmup", "prewarm") or \
                         await _call_safe("warmup", "warm")

                if _is_ready(svc):
                    info["ready"] = True
                    break

                raise RuntimeError("vector service not ready after warmup")

            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"
                if retries >= self._max_retries:
                    break
                delay = self._retry_backoff_ms / 1000.0
                delay *= (1.0 + random.random() * 0.25)
                await asyncio.sleep(delay)
                retries += 1

        info["ms"] = int((time.perf_counter() - t0) * 1000)
        info["post_status"] = _get_status(svc)
        if info["ready"]:
            self._dump_debug_artifacts(name, svc, info["post_status"])
        return info


    async def start(self) -> Dict[str, Any]:
        if self._started:
            return {"already_started": True, **self.status}
        self._started = True
        self._error = None
        self._finished = False
        self._t0 = time.perf_counter()

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

            if not tasks:
                self._t1 = time.perf_counter()
                self._finished = True
                self._last_summary = {}
                return {"ready": True, "summary": {}, **self.status}

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=False),
                timeout=self._timeout_ms / 1000.0
            )
            self._t1 = time.perf_counter()
            self._finished = True

            summary = {
                r["service"]: {
                    "ready": r["ready"],
                    "attempts": r["attempts"],
                    "ms": r["ms"],
                    "error": r["error"],
                    # NEW: snapshots for diffing / debugging
                    "pre_status": r.get("pre_status"),
                    "post_status": r.get("post_status"),
                }
                for r in results
            }
            self._last_summary = summary

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

        

    def _dump_debug_artifacts(self, name: str, svc: Any, status_dict: Dict[str, Any]) -> None:
        """Best-effort: write status JSON and optional .npy arrays for inspection."""
        if not self._debug_dir:
            return

        safe_name = name.replace("/", "_").replace(" ", "_")
        base = os.path.join(self._debug_dir, safe_name)

        # 1) status JSON
        try:
            with open(base + ".status.json", "w", encoding="utf-8") as f:
                json.dump(status_dict or {}, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.debug("vector %s: failed to write status.json: %s", name, e)

        # 2) service-provided dump method (if available)
        try:
            if _has(svc, "dump_debug_artifacts"):
                svc.dump_debug_artifacts(self._debug_dir)
                return
        except Exception as e:
            logger.debug("vector %s.dump_debug_artifacts failed: %s", name, e)

        # 3) optional .npy dumps (embeddings, index vectors, etc.)
        if not DEBUG_DUMP_NUMPY or np is None:
            return

        candidate_attrs = (
            "embeddings",
            "embedding_matrix",
            "index_vectors",
            "_embeddings",
            "_embedding_matrix",
        )
        for attr in candidate_attrs:
            try:
                arr = getattr(svc, attr, None)
                if arr is None:
                    continue
                # tolerate torch.Tensor as well
                if hasattr(arr, "cpu") and hasattr(arr, "numpy"):
                    arr = arr.cpu().numpy()
                if not isinstance(arr, (list, tuple)) and hasattr(arr, "shape"):
                    np.save(base + f".{attr}.npy", arr)
                    logger.info("vector %s: dumped %s to %s.%s.npy (shape=%s)",
                                name, attr, base, attr, getattr(arr, "shape", None))
            except Exception as e:
                logger.debug("vector %s: failed to dump attr=%s as npy: %s", name, attr, e)

