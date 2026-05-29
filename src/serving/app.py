"""FastAPI application factory and lifespan handler."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.serving import model_loader
from src.serving.router import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup via the dedicated MLX executor thread."""
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(model_loader.get_mlx_executor(), model_loader.load_model)
    except FileNotFoundError as exc:
        logger.warning("Model not loaded on startup: %s", exc)
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Intent Classification API",
        description="LLM-based e-commerce customer support intent classifier",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
