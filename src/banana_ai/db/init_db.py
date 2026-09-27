from banana_ai.db.session import Base, engine
from banana_ai.db import models  # noqa: F401


def main():
    # create_all is intentionally simple for this learning project.
    # For production, replace this with Alembic migrations.
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")


if __name__ == "__main__":
    main()
