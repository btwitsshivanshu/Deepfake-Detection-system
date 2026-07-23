"""Auth service: registration, login, audit logging."""
from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.models.audit_log import AuditLog
from app.models.user import User
from app.schemas.schemas import Token, UserCreate, UserLogin, UserOut


def _log(db: Session, user_id: int | None, action: str, ip: str | None = None) -> None:
    db.add(AuditLog(user_id=user_id, action=action, entity="user", ip_address=ip))
    db.commit()


def register(db: Session, payload: UserCreate, ip: str | None = None) -> UserOut:
    exists = (
        db.query(User)
        .filter((User.email == payload.email) | (User.username == payload.username))
        .first()
    )
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "Username or email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _log(db, user.id, "register", ip)
    return UserOut.model_validate(user)


def login(db: Session, payload: UserLogin, ip: str | None = None) -> Token:
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account disabled")

    _log(db, user.id, "login", ip)
    return Token(
        access_token=create_access_token(user.email, user.role),
        refresh_token=create_refresh_token(user.email),
        user=UserOut.model_validate(user),
    )
