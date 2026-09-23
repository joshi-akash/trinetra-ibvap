"""
TRINETRA Database Session Provider
Re-exports database engine, SessionLocal, and get_db conforming to PDF Section 5.3.3 manifest:
backend/app/database/session.py
"""
from backend.app.database import engine, SessionLocal, get_db, init_db

__all__ = ["engine", "SessionLocal", "get_db", "init_db"]
