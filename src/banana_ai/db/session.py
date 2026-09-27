from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from banana_ai.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM tables."""
    pass


# SQLAlchemy manages the connection pool for us.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db():
    """FastAPI dependency that opens/closes a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
