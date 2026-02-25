from pydantic import BaseModel


class PredictionInput(BaseModel):
    temperature: float
    humidity: float
    soil_moisture: float
    rainfall: float