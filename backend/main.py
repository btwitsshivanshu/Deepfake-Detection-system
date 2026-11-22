"""
FastAPI application entry point (Phase 8).

Run from the `backend/` directory:
    uvicorn main:app --reload

Requires the project root on PYTHONPATH so the `ml` package is importable.
`main.py` inserts it automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure both `backend/app` and project root (for `ml`) are importable.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
for p in (BACKEND_DIR, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI                       # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles       # noqa: E402

from app.config import settings                    # noqa: E402
from app.database.session import init_db           # noqa: E402
from app.routers import admin, analytics, auth, detection, history  # noqa: E402


def create_app() -> FastAPI:
    app = FastAPI(
        title="AuthentiScan API",
        description="AuthentiScan — detect deepfake images & videos (GAN faces, face swaps, AI-generated images) with an explainable ensemble.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve generated heatmaps / uploads for the frontend.
    Path(settings.RESULT_DIR).mkdir(parents=True, exist_ok=True)
    app.mount("/storage/results",
              StaticFiles(directory=settings.RESULT_DIR), name="results")

    for r in (auth.router, detection.router, history.router,
              analytics.router, admin.router):
        app.include_router(r)

    @app.on_event("startup")
    def _startup() -> None:
        # Tables are primarily managed by database/schema.sql; this is a safety net.
        try:
            init_db()
        except Exception as exc:  # pragma: no cover - non-fatal in dev
            print(f"[startup] init_db skipped: {exc}")

        # Warm up the ensemble (3 detectors + CLIP) in the background so the
        # first user request doesn't wait ~20 s for model loading.
        if settings.DEFAULT_MODEL == "ensemble":
            import threading

            def _warm() -> None:
                try:
                    from ml.ensemble import _load
                    _load()
                    print("[startup] ensemble models loaded")
                except Exception as exc:  # pragma: no cover
                    print(f"[startup] ensemble warm-up skipped: {exc}")

            threading.Thread(target=_warm, daemon=True).start()

    @app.get("/api/health", tags=["Health"])
    def health() -> dict:
        return {"status": "ok", "service": "deepfake-detection", "version": "1.0.0"}

    return app


app = create_app()
