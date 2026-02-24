from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# CHANGE THIS TO YOUR MYSQL SETTINGS
DATABASE_URL = "mysql+pymysql://root:12345@localhost/irrigation_db"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()