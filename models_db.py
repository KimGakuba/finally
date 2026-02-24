from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True)
    password = Column(String(200))


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    prediction = Column(String(50))
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)