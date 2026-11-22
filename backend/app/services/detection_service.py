"""
Detection service — orchestrates upload → inference → persistence.

Bridges the FastAPI layer and the ML inference engine (ml.inference),
saving uploaded files, running the selected model, generating Grad-CAM,
and persisting Prediction + DetectionResult rows.
"""
from __future__ import annotations

import io
import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from starlette.datastructures import Headers

from app.config import settings
from app.models.audit_log import AuditLog
from app.models.ml_model import MLModel
from app.models.prediction import DetectionResult, Prediction
from app.models.uploaded_file import UploadedFile
from app.models.user import User
from app.schemas.schemas import DetectionResponse, FrameResult

# ML engine (imported lazily-safe: package `ml` must be on PYTHONPATH)
from ml.inference import predict_image, predict_video

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def _resolve_model(db: Session, architecture: str | None) -> MLModel:
    arch = architecture or settings.DEFAULT_MODEL
    model = (
        db.query(MLModel)
        .filter(MLModel.architecture == arch, MLModel.is_active.is_(True))
        .first()
    )
    if not model:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"No active model for architecture '{arch}'")
    return model


def _save_upload(upload: UploadFile, user_id: int) -> tuple[Path, str, int]:
    ext = Path(upload.filename or "").suffix.lower()
    if ext in IMAGE_EXT:
        media_type = "image"
    elif ext in VIDEO_EXT:
        media_type = "video"
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Unsupported file type: {ext}")

    dest_dir = Path(settings.UPLOAD_DIR) / str(user_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}{ext}"

    with dest.open("wb") as buf:
        shutil.copyfileobj(upload.file, buf)

    size = dest.stat().st_size
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if size > max_bytes:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"File exceeds {settings.MAX_UPLOAD_MB} MB limit")
    return dest, media_type, size


def _to_url(path: str | None) -> str | None:
    if not path:
        return None
    return "/" + str(Path(path).as_posix()).lstrip("./")


def run_detection(
    db: Session,
    user: User,
    upload: UploadFile,
    architecture: str | None,
) -> DetectionResponse:
    model = _resolve_model(db, architecture)
    dest, media_type, size = _save_upload(upload, user.id)

    file_row = UploadedFile(
        user_id=user.id, filename=upload.filename or dest.name,
        storage_path=str(dest), media_type=media_type,
        mime_type=upload.content_type, size_bytes=size,
    )
    db.add(file_row)
    db.flush()  # get file_row.id without full commit

    try:
        if media_type == "image":
            result = predict_image(dest, model.architecture,
                                   model.weights_path, settings.RESULT_DIR)
        else:
            result = predict_video(dest, model.architecture,
                                   model.weights_path, settings.RESULT_DIR)
    except FileNotFoundError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Model weights not found. Train the model first.")
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    prediction = Prediction(
        user_id=user.id, file_id=file_row.id, model_id=model.id,
        media_type=media_type, verdict=result["verdict"],
        confidence=result["confidence"], frames_analyzed=result["frames_analyzed"],
        inference_ms=result.get("inference_ms"),
    )
    db.add(prediction)
    db.flush()

    for fr in result["frames"]:
        db.add(DetectionResult(
            prediction_id=prediction.id,
            frame_index=fr["frame_index"],
            fake_probability=fr["fake_probability"],
            heatmap_path=fr.get("heatmap_path") or result.get("heatmap_path"),
        ))

    details = result["verdict"] + (f" ({result['fake_type']})" if result.get("fake_type") else "")
    db.add(AuditLog(user_id=user.id, action="detection",
                    entity=media_type, details=details))
    db.commit()
    db.refresh(prediction)

    return DetectionResponse(
        prediction_id=prediction.id,
        verdict=result["verdict"],
        confidence=result["confidence"],
        fake_probability=result["fake_probability"],
        media_type=media_type,
        frames_analyzed=result["frames_analyzed"],
        inference_ms=result.get("inference_ms"),
        heatmap_url=_to_url(result.get("heatmap_path")),
        model_used=model.name,
        fake_type=result.get("fake_type"),
        detector_scores=result.get("detector_scores"),
        screenshot_crop=result.get("screenshot_crop"),
        frames=[FrameResult(**fr) for fr in result["frames"]],
    )


def compare_all_models(
    db: Session, user: User, upload: UploadFile
) -> list[DetectionResponse]:
    """Run every active model on the same file (Feature F6)."""
    architectures = [m.architecture for m in
                     db.query(MLModel).filter(MLModel.is_active.is_(True)).all()]

    # Read the upload once, then feed a fresh in-memory copy to each run.
    content = upload.file.read()
    filename, content_type = upload.filename, upload.content_type

    responses: list[DetectionResponse] = []
    for arch in architectures:
        fresh = UploadFile(
            filename=filename,
            file=io.BytesIO(content),
            headers=Headers({"content-type": content_type or ""}),
        )
        responses.append(run_detection(db, user, fresh, arch))
    return responses
