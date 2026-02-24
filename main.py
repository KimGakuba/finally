from fastapi import FastAPI
from database import engine
from models_db import Base
from routes.predict import router

app = FastAPI(title="Smart Irrigation DSS")

# create database tables automatically
Base.metadata.create_all(bind=engine)

# include routes
app.include_router(router)