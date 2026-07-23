"""Analytics & model-listing routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.models.ml_model import MLModel
from app.models.user import User
from app.schemas.schemas import AnalyticsSummary, ModelOut
from app.services import analytics_service

router = APIRouter(prefix="/api", tags=["Analytics"])


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return analytics_service.summary(db, user)


@router.get("/models", response_model=list[ModelOut])
def list_models(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return db.query(MLModel).filter(MLModel.is_active.is_(True)).all()
