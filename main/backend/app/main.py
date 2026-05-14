"""
Healthcare DL — FastAPI Application Entry Point

Startup lifecycle:
1. Load all 7 models into memory (GPU if available)
2. Set models to eval() mode
3. Run warm-up inference
4. Mark app as ready

CORS is configured to allow the Vite dev server and production frontend.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.model_registry import registry
from app.routes import health, predict, report
from app.utils.logger import logger


# ── Startup / Shutdown lifecycle ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info("=" * 50)
    logger.info("Healthcare DL API starting up...")
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"GPU enabled: {settings.USE_GPU}")
    logger.info("=" * 50)

    # Load all models
    registry.load_all_models()

    logger.info("=" * 50)
    logger.info("Server is ready to accept requests")
    logger.info("=" * 50)

    yield  # App runs here

    # Shutdown
    logger.info("Server shutting down...")


# ── Create FastAPI app ──
app = FastAPI(
    title="Healthcare DL API",
    description="AI-powered medical image analysis — Zero-choice UX",
    version=settings.VERSION,
    lifespan=lifespan,
)


# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,        # Configured production URL (or localhost in dev)
        "http://localhost:5173",       # Vite dev server fallback
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",  # All Vercel preview deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register Routes ──
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(report.router, tags=["Report"])


# ── Global Exception Handler ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return clean JSON errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    if settings.ENV == "development":
        detail = str(exc)
    else:
        detail = "An internal server error occurred. Please try again."

    return JSONResponse(
        status_code=500,
        content={"detail": detail},
    )
