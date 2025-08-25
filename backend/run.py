# backend/run.py
import logging
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from app.api.router import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"  # ../frontend

def create_app() -> FastAPI:
    app = FastAPI(title="HCM AI Portal API", version="0.2")

    # CORS (adjust for prod)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes (they live under /api)
    app.include_router(api_router)

    # Serve static frontend
    if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
        # Mount static assets (CSS, JS, images, etc.) - if you have an assets folder
        assets_dir = FRONTEND_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        
        # Serve index.html for root and handle client-side routing
        @app.get("/", include_in_schema=False)
        def serve_frontend():
            return FileResponse(str(FRONTEND_DIR / "index.html"))
        
        # Handle client-side routing - serve index.html for non-API routes
        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_frontend_routes(full_path: str):
            # Don't intercept API routes
            if full_path.startswith("api/"):
                return {"detail": "Not Found"}
            
            # For any other route, serve the frontend
            index_file = FRONTEND_DIR / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
            return RedirectResponse(url="/docs")
    else:
        # Fallback if no frontend exists
        @app.get("/", include_in_schema=False)
        def home_fallback():
            return RedirectResponse(url="/docs")

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)