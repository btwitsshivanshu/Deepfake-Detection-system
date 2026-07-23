"""Detection routes: image, video and multi-model comparison."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.models.user import User
from app.schemas.schemas import DetectionResponse
from app.services import detection_service

router = APIRouter(prefix="/api/detect", tags=["Detection"])


@router.post("/image", response_model=DetectionResponse)
def detect_image(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return detection_service.run_detection(db, user, file, model)


@router.post("/video", response_model=DetectionResponse)
def detect_video(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return detection_service.run_detection(db, user, file, model)


@router.post("/compare", response_model=list[DetectionResponse])
def compare_models(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return detection_service.compare_all_models(db, user, file)
