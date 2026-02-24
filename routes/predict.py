from fastapi import APIRouter
from schemas import PredictionInput
from services.prediction_service import predict_irrigation

router = APIRouter()


@router.post("/predict")
def predict(data: PredictionInput):

    result = predict_irrigation(data.dict())

    return result