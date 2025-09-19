# backend/app/vector_bootstrap.py
import os
import time
import asyncio
import logging
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_STARTUP_WARM = os.getenv("VECTOR_STARTUP_WARM", "true").lower() == "true"
DEFAULT_STARTUP_TIMEOUT = int(os.getenv("VECTOR_STARTUP_TIMEOUT_MS", "60000"))  # 60s
DEFAULT_RETRY_BACKOFF_MS = int(os.getenv("VECTOR_STARTUP_RETRY_BACKOFF_MS", "800"))
DEFAULT_MAX_RETRIES = int(os.getenv("VECTOR_STARTUP_MAX_RETRIES", "3"))

def _has(obj: Any, name: str) -> bool:
    return hasattr(obj, name) and callable(getattr(obj, name, None))

async def _maybe_async_call(fn: Callable, *args, **kwargs):
    """Call sync or async function uniformly in asyncio context."""
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

class VectorBootstrapper:
    """
    Proactively builds/loads vector indices at startup (and on demand).
    This class tolerates different vector service implementations by probing
    for common methods and falling back if they don't exist.
    """
    def __init__(
        self,
        nlp_services: Dict[str, Any],
        *,
        warm_on_start: bool = DEFAULT_STARTUP_WARM,
        startup_timeout_ms: int = DEFAULT_STARTUP_TIMEOUT,
        retry_backoff_ms: int = DEFAULT_RETRY_BACKOFF_MS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._nlp_services = nlp_services  # {"primary": obj, "original": obj?, "language_native": obj?}
        self._warm_on_start = warm_on_start
        self._timeout_ms = startup_timeout_ms
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
            "max_retries": self._max_retries,
        }

    async def _ensure_one_ready(self, name: str, svc: Any) -> Dict[str, Any]:
        """
        Make a single vector service ready. We detect & call:
          - ensure_index_loaded() or build_or_load_index() or load()
          - warmup() or warm(query=...) or prewarm()
          - vector_status() for readiness check
        All are optional; we try what's available.
        """
        info = {"service": name, "attempts": 0, "ready": False, "error": None, "ms": 0}
        t0 = time.perf_counter()

        def _ready(s: Any) -> bool:
            try:
                if _has(s, "vector_status"):
                    st = s.vector_status()
                    return bool(st.get("ready"))
            except Exception:
                # If vector_status is not reliable, consider presence of an index object
                pass
            return False

        async def _build_or_load(s: Any):
            if _has(s, "ensure_index_loaded"):
                return await _maybe_async_call(s.ensure_index_loaded)
            if _has(s, "build_or_load_index"):
                return await _maybe_async_call(s.build_or_load_index)
            if _has(s, "load"):
                return await _maybe_async_call(s.load)
            # no build/load API, skip

        async def _warm(s: Any):
            # Some services provide warmup routines
            if _has(s, "warmup"):
                return await _maybe_async_call(s.warmup)
            if _has(s, "prewarm"):
                return await _maybe_async_call(s.prewarm)
            if _has(s, "warm"):
                return await _maybe_async_call(s.warm)
            # Fallback: optional small similarity probe if method exists
            if _has(s, "search"):
                try:
                    # Non-fatal, quick probe
                    await _maybe_async_call(s.search, "healthcheck", top_k=1)
                except Exception:
                    pass

        # If it's already ready, we're done
        try:
            if _ready(svc):
                info["ready"] = True
                info["ms"] = int((time.perf_counter() - t0) * 1000)
                return info
        except Exception as e:
            logger.debug("vector_status probe failed for %s: %s", name, e)

        # Else, build & warm with retries
        retries = 0
        while retries <= self._max_retries:
            try:
                info["attempts"] += 1
                await _build_or_load(svc)
                await _warm(svc)
                if _ready(svc):
                    info["ready"] = True
                    break
                # If still not ready, raise to enter retry flow
                raise RuntimeError("Vector service not ready after warmup")
            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"
                if retries >= self._max_retries:
                    break
                await asyncio.sleep(self._retry_backoff_ms / 1000.0)
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

        if not self._warm_on_start:
            self._finished = True
            self._t1 = time.perf_counter()
            return {"skipped": True, **self.status}

        try:
            tasks = []
            for name, svc in self._nlp_services.items():
                if not svc:
                    continue
                tasks.append(self._ensure_one_ready(name, svc))
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=False),
                                             timeout=self._timeout_ms / 1000.0)
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
