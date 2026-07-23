"""Admin routes: user management, audit logs (admin-only)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import require_admin
from app.database.session import get_db
from app.models.audit_log import AuditLog
from app.models.user import User
from app.schemas.schemas import UserOut

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    return db.query(User).order_by(User.created_at.desc()).all()


@router.patch("/users/{user_id}/toggle", response_model=UserOut)
def toggle_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    user = db.query(User).get(user_id)
    if user:
        user.is_active = not user.is_active
        db.commit()
        db.refresh(user)
    return user


@router.get("/audit-logs")
def audit_logs(
    limit: int = 100,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    return (
        db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()
    )
