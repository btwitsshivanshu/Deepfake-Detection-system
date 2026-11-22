"""Prediction and per-frame DetectionResult ORM models."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    file_id: Mapped[int] = mapped_column(ForeignKey("uploaded_files.id", ondelete="CASCADE"))
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id", ondelete="RESTRICT"))
    media_type: Mapped[str] = mapped_column(Enum("image", "video"), nullable=False)
    verdict: Mapped[str] = mapped_column(Enum("real", "fake", "uncertain"), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    frames_analyzed: Mapped[int] = mapped_column(Integer, default=1)
    inference_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="predictions")
    file = relationship("UploadedFile", back_populates="predictions")
    results = relationship("DetectionResult", back_populates="prediction",
                           cascade="all, delete-orphan")


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"))
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    fake_probability: Mapped[float] = mapped_column(Float, nullable=False)
    heatmap_path: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    prediction = relationship("Prediction", back_populates="results")
