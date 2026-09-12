import hmac
import hashlib
import json
import base64
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# --- Password Hashing (PBKDF2-SHA256) ---
def get_password_hash(password: str) -> str:
    salt = secrets.token_hex(16)
    iterations = 100000
    hash_bytes = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        iterations
    )
    return f"pbkdf2:sha256:{iterations}${salt}${hash_bytes.hex()}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        parts = hashed_password.split("$")
        if len(parts) != 3:
            return False
        meta, salt, original_hash = parts
        iterations = int(meta.split(":")[-1])
        calculated_hash = hashlib.pbkdf2_hmac(
            'sha256',
            plain_password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations
        ).hex()
        return hmac.compare_digest(original_hash, calculated_hash)
    except Exception:
        return False

# --- Standard JWT Token Creation & Verification ---
def _b64encode_url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

def _b64decode_url(s: str) -> bytes:
    padding = '=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = int(expire.timestamp())

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64encode_url(json.dumps(header, separators=(',', ':')).encode('utf-8'))
    payload_b64 = _b64encode_url(json.dumps(to_encode, separators=(',', ':')).encode('utf-8'))

    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    signature = hmac.new(settings.JWT_SECRET.encode('utf-8'), signing_input, hashlib.sha256).digest()
    signature_b64 = _b64encode_url(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"

def decode_access_token(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed token")
        header_b64, payload_b64, signature_b64 = parts

        signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
        expected_sig = hmac.new(settings.JWT_SECRET.encode('utf-8'), signing_input, hashlib.sha256).digest()
        actual_sig = _b64decode_url(signature_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Signature verification failed")

        payload_bytes = _b64decode_url(payload_b64)
        payload = json.loads(payload_bytes.decode('utf-8'))

        # Expiration check
        exp = payload.get("exp")
        if exp and datetime.now(timezone.utc).timestamp() > exp:
            raise ValueError("Token has expired")

        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- Role-Based Access Dependencies ---
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject identifier",
            headers={"WWW-Authenticate": "Bearer"}
        )
    user = db.query(User).filter(User.id == user_id, User.active == True).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user

def require_roles(allowed_roles: List[str]):
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required role: {allowed_roles}, your role: {current_user.role}"
            )
        return current_user
    return role_checker
