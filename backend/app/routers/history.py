"""History routes: list and fetch predictions."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.models.user import User
from app.schemas.schemas import PredictionOut
from app.services import history_service

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=list[PredictionOut])
def list_history(
    verdict: str | None = Query(default=None),
    media_type: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return history_service.list_history(db, user, verdict, media_type, limit, offset)


@router.get("/{prediction_id}", response_model=PredictionOut)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return history_service.get_prediction(db, user, prediction_id)
