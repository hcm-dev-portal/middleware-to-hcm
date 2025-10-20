# run.py  (only the changed/added lines are marked with <<< >>>)

import os
import logging
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from app.api.speech_routes import speech_router
from app.api.router import router as api_router

# ✅ DB + NLP
from app.services.db_service import SQLServerDatabaseService, set_request_id
from app.services.nlp_service_2 import LanguageNativeNLPService

# ✅ ADD: reports routers (API + static)
# <<< add
from app.api.router import (
    router as reports_router,
    static_router as reports_static_router,
)
# >>>

try:
    from app.vector_bootstrap import VectorBootstrapper
except Exception:
    VectorBootstrapper = None

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app.http")

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
STATIC_DIR   = FRONTEND_DIR / "static"
ASSETS_DIR   = FRONTEND_DIR / "assets"
LANG_DIR     = FRONTEND_DIR / "lang"
INDEX_HTML   = FRONTEND_DIR / "index.html"

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    try:
        db = SQLServerDatabaseService()
        app.state.db = db

        nlp = LanguageNativeNLPService(db_service=db)
        app.state.nlp = nlp

        warm_timeout_s = int(os.getenv("VECTOR_WARMUP_TIMEOUT_S", "20"))
        warm_block = os.getenv("VECTOR_WARMUP_BLOCKING", "true").lower() == "true"

        if VectorBootstrapper:
            app.state.vector_bootstrap = VectorBootstrapper({"primary": getattr(app.state, "nlp", None)})
            try:
                if warm_block:
                    logger.info("Vector warmup (blocking, timeout=%ss) starting ...", warm_timeout_s)
                    result = await app.state.vector_bootstrap.start()
                    logger.info("Vector warmup completed: %s", result)
                else:
                    import asyncio
                    logger.info("Vector warmup (background) scheduled.")
                    asyncio.create_task(app.state.vector_bootstrap.start(timeout_s=warm_timeout_s))
            except Exception as e:
                logger.exception("Vector warmup error: %s", e)
        else:
            try:
                if hasattr(nlp, "vector_search") and hasattr(nlp.vector_search, "warmup"):
                    logger.info("Vector warmup via NLP service starting ...")
                    maybe = nlp.vector_search.warmup()
                    if hasattr(maybe, "__await__"):
                        import asyncio
                        await asyncio.wait_for(maybe, timeout=warm_timeout_s)
                    logger.info("Vector warmup via NLP service completed.")
            except Exception as e:
                logger.exception("Vector warmup (fallback) error: %s", e)

        logger.info("Startup complete in %dms", int((time.perf_counter() - t0) * 1000))
    except Exception as e:
        logger.exception("Service initialization failed: %s", e)

    try:
        yield
    finally:
        logger.info("Shutting down services ...")

def create_app() -> FastAPI:
    app = FastAPI(
        title="HCM AI Portal API",
        version=os.getenv("APP_VERSION", "0.4"),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def rid_and_access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        set_request_id(request_id)
        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            dur_ms = int((time.perf_counter() - start) * 1000)
            status = getattr(response, "status_code", "?")
            logger.info("HTTP %s %s -> %s rid=%s dur=%dms",
                        request.method, request.url.path, status, request_id, dur_ms)
            try:
                if response is not None:
                    response.headers["x-request-id"] = request_id
            except Exception:
                pass

    # CORS (configurable)
    raw_origins = os.getenv("CORS_ORIGINS", "*").strip()
    allow_origins = ["*"] if raw_origins == "*" else [o.strip() for o in raw_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- Static mounts ----------
    # Frontend assets
    if FRONTEND_DIR.exists():
        if ASSETS_DIR.exists():
            app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        if LANG_DIR.exists():
            app.mount("/lang", StaticFiles(directory=str(LANG_DIR)), name="lang")

        LEAVE_HTML = FRONTEND_DIR / "leave_page.html"
        if LEAVE_HTML.exists():
            @app.get("/leave", include_in_schema=False)
            async def leave_page():
                return FileResponse(str(LEAVE_HTML))

        @app.get("/translations.js", include_in_schema=False)
        async def serve_translations_js():
            cand = FRONTEND_DIR / "translations.js"
            if cand.exists():
                return FileResponse(str(cand))
            cand2 = STATIC_DIR / "translations.js"
            if cand2.exists():
                return FileResponse(str(cand2))
            return Response(status_code=404)

        if INDEX_HTML.exists():
            @app.get("/", include_in_schema=False)
            async def root_index():
                return FileResponse(str(INDEX_HTML))
        logger.info("Frontend dir: %s (exists=%s)", FRONTEND_DIR, True)
    else:
        logger.warning("Frontend dir not found at %s", FRONTEND_DIR)

    # Charts/images mount (for VisualizationService outputs)
    charts_dir = REPO_ROOT / "charts" / "images"
    try:
        charts_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception("Failed to ensure charts dir exists: %s", e)
    if charts_dir.exists():
        app.mount("/charts/images", StaticFiles(directory=str(charts_dir)), name="charts_images")
        logger.info("Mounted charts at /charts/images -> %s", charts_dir)
    else:
        logger.warning("Charts dir not found at %s; /charts/images will 404", charts_dir)

    # ---------- API routes ----------
    app.include_router(api_router)
    app.include_router(speech_router)

    # Reports API and static page(s)
    app.include_router(reports_router)          # /api/reports/...
    app.include_router(reports_static_router)   # /generate_report.html

    return app



app = create_app()

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8899"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run(app, host=host, port=port, reload=reload)
