# backend/run.py
import logging
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from starlette.concurrency import run_in_threadpool

from app.api.router import router as api_router
from app.services.db_service import set_request_id, SQLServerDatabaseService
from app.services.nlp_service import NLPService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("app.http")

BASE_DIR = Path(__file__).resolve().parent         # backend/
FRONTEND_DIR = BASE_DIR.parent / "frontend"
STATIC_DIR   = FRONTEND_DIR / "static"
ASSETS_DIR   = FRONTEND_DIR / "assets"
LANG_DIR     = FRONTEND_DIR / "lang"       # <repo-root>/frontend

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create long-lived services once and store them on app.state."""
    try:
        db = SQLServerDatabaseService()
        nlp = NLPService(db_service=db)
        app.state.db = db
        app.state.nlp = nlp
        logger.info("App services initialized: db, nlp")
    except Exception as e:
        logger.exception(f"Service init failed: {type(e).__name__}: {e}")
    try:
        yield
    finally:
        logger.info("App services shutting down")

def create_app() -> FastAPI:
    app = FastAPI(title="HCM AI Portal API", version="0.2", lifespan=lifespan)

    # Request ID + access logging
    @app.middleware("http")
    async def rid_and_access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        set_request_id(rid)
        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            dur_ms = int((time.perf_counter() - start) * 1000)
            status = getattr(response, "status_code", "?")
            logger.info("HTTP %s %s -> %s rid=%s dur=%dms",
                        request.method, request.url.path, status, rid, dur_ms)
            try:
                if response is not None:
                    response.headers["x-request-id"] = rid
            except Exception:
                pass

    # CORS (wide open for dev)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static assets for SPA (optional convenience; router also serves "/")
    if FRONTEND_DIR.exists():
        assets = FRONTEND_DIR / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")
        index_file = FRONTEND_DIR / "index.html"

        # Static mounts
        if ASSETS_DIR.exists():
            app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        if LANG_DIR.exists():
            app.mount("/lang", StaticFiles(directory=str(LANG_DIR)), name="lang")
            # Serve translations.js at /translations.js (root of frontend or inside /static)
            @app.get("/translations.js", include_in_schema=False)
            async def serve_translations_js():
                cand = FRONTEND_DIR / "translations.js"
                if cand.exists():
                    return FileResponse(str(cand))
                cand2 = STATIC_DIR / "translations.js"
                if cand2.exists():
                    return FileResponse(str(cand2))
                return Response(status_code=404)

        if index_file.exists():
            @app.get("/", include_in_schema=False)
            async def root_index():
                # If you prefer the router's "/" handler, delete this endpoint.
                return FileResponse(str(index_file))
        logger.info(f"Frontend dir: {FRONTEND_DIR} (exists={FRONTEND_DIR.exists()})")
    else:
        logger.warning(f"Frontend dir not found at {FRONTEND_DIR}")

    # API routes (all endpoints live in app/api/router.py)
    app.include_router(api_router)

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Pass the app object to avoid module-name confusion
    uvicorn.run(app, host="0.0.0.0", port=8899, reload=False)
