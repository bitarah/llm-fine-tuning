"""Pydantic request/response models for the serving API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    approach: str = Field(default="fine_tuned")  # zero_shot | few_shot | fine_tuned


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    approach_used: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    base_model_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
