"""Analytics service — aggregate a user's prediction statistics."""
from __future__ import annotations

from collections import Counter

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.ml_model import MLModel
from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.schemas import AnalyticsSummary


def summary(db: Session, user: User) -> AnalyticsSummary:
    rows = db.query(Prediction).filter(Prediction.user_id == user.id).all()
    total = len(rows)

    verdicts = Counter(r.verdict for r in rows)
    media = Counter(r.media_type for r in rows)
    avg_conf = (sum(r.confidence for r in rows) / total) if total else 0.0

    # Model usage by name
    model_names = {m.id: m.name for m in db.query(MLModel).all()}
    model_usage = Counter(model_names.get(r.model_id, "unknown") for r in rows)

    # Daily counts (YYYY-MM-DD)
    daily = Counter(r.created_at.strftime("%Y-%m-%d") for r in rows if r.created_at)

    return AnalyticsSummary(
        total_predictions=total,
        real_count=verdicts.get("real", 0),
        fake_count=verdicts.get("fake", 0),
        uncertain_count=verdicts.get("uncertain", 0),
        image_count=media.get("image", 0),
        video_count=media.get("video", 0),
        avg_confidence=round(avg_conf, 4),
        model_usage=dict(model_usage),
        daily_counts=dict(sorted(daily.items())),
    )
