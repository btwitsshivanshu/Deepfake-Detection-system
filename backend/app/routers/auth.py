"""Authentication routes: register, login, current-user profile."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database.session import get_db
from app.models.user import User
from app.schemas.schemas import Token, UserCreate, UserLogin, UserOut
from app.services import auth_service

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserCreate, request: Request, db: Session = Depends(get_db)):
    return auth_service.register(db, payload, request.client.host if request.client else None)


@router.post("/login", response_model=Token)
def login(payload: UserLogin, request: Request, db: Session = Depends(get_db)):
    return auth_service.login(db, payload, request.client.host if request.client else None)


@router.post("/token", response_model=Token, include_in_schema=False)
def login_form(
    form: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """OAuth2-compatible endpoint so Swagger 'Authorize' works (username=email)."""
    payload = UserLogin(email=form.username, password=form.password)
    return auth_service.login(db, payload, request.client.host if request and request.client else None)


@router.get("/me", response_model=UserOut)
def me(current: User = Depends(get_current_user)):
    return current
