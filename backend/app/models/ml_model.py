"""ML model registry ORM model."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class MLModel(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    architecture: Mapped[str] = mapped_column(String(50), nullable=False)
    weights_path: Mapped[str] = mapped_column(String(255), nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float)
    auc: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
