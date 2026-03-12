from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")

connect_args: dict[str, object] = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    connect_args = {"sslmode": "require"}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

Base = declarative_base()
