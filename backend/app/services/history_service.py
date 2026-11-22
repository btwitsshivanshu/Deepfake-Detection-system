"""History service — list & fetch a user's predictions."""
from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.prediction import Prediction
from app.models.user import User


def list_history(
    db: Session,
    user: User,
    verdict: str | None = None,
    media_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Prediction]:
    query = db.query(Prediction).filter(Prediction.user_id == user.id)
    if verdict in {"real", "fake", "uncertain"}:
        query = query.filter(Prediction.verdict == verdict)
    if media_type in {"image", "video"}:
        query = query.filter(Prediction.media_type == media_type)
    return (
        query.order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(min(limit, 200))
        .all()
    )


def get_prediction(db: Session, user: User, prediction_id: int) -> Prediction:
    pred = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == user.id)
        .first()
    )
    if not pred:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prediction not found")
    return pred
