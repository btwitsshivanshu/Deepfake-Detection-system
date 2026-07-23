"""Pydantic schemas (DTOs) for request/response validation."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------- Auth ----------
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------- Models ----------
class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    architecture: str
    accuracy: float | None = None
    auc: float | None = None
    f1_score: float | None = None
    is_active: bool


# ---------- Detection ----------
class FrameResult(BaseModel):
    frame_index: int
    fake_probability: float
    heatmap_path: str | None = None


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    media_type: str
    verdict: str
    confidence: float
    frames_analyzed: int
    inference_ms: int | None = None
    created_at: datetime


class DetectionResponse(BaseModel):
    prediction_id: int
    verdict: str
    confidence: float
    fake_probability: float
    media_type: str
    frames_analyzed: int
    inference_ms: int | None = None
    heatmap_url: str | None = None
    model_used: str
    frames: list[FrameResult] = []


# ---------- Analytics ----------
class AnalyticsSummary(BaseModel):
    total_predictions: int
    real_count: int
    fake_count: int
    image_count: int
    video_count: int
    avg_confidence: float
    model_usage: dict[str, int]
    daily_counts: dict[str, int]
