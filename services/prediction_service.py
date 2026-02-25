import joblib
import numpy as np
import os
import gdown
from services.preprocessing import preprocess_input
from services.business_logic import apply_business_rules
from database import SessionLocal
from models_db import PredictionLog

MODEL_PATH = "models/smart_irrigation_model.pkl"
ENCODER_PATH = "models/target_encoder.pkl"

os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/file/d/1MvqPRk7KnfO9GNVpftKOCvtsyLO8AnHQ/view?usp=drive_link",
        MODEL_PATH,
        quiet=False,
        fuzzy=True
    )

if not os.path.exists(ENCODER_PATH):
    gdown.download(
        "https://drive.google.com/file/d/13dCZ-773wciiFKQjua-Q2dcoHNnfScZt/view?usp=drive_link",
        ENCODER_PATH,
        quiet=False,
        fuzzy=True
    )

model = joblib.load(MODEL_PATH)
target_encoder = joblib.load(ENCODER_PATH)

def save_log(prediction, confidence):
    db = SessionLocal()
    log = PredictionLog(
        user_id=1,
        prediction=prediction,
        confidence=confidence
    )
    db.add(log)
    db.commit()
    db.close()

def predict_irrigation(data):
    df = preprocess_input(data)
    probs = model.predict_proba(df)
    pred_class = np.argmax(probs, axis=1)[0]
    confidence = float(np.max(probs) * 100)
    label = target_encoder.inverse_transform([pred_class])[0]
    recommendation = apply_business_rules(label, data["rainfall"])
    save_log(label, confidence)

    if confidence > 90:
        color = "green"
    elif confidence > 80:
        color = "yellow"
    else:
        color = "red"

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "color": color,
        "recommendation": recommendation
    }
