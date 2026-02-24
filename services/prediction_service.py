import joblib
import numpy as np

from services.preprocessing import preprocess_input
from services.business_logic import apply_business_rules

from database import SessionLocal
from models_db import PredictionLog

# load model
model = joblib.load("models/smart_irrigation_model.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")


def save_log(prediction, confidence):

    db = SessionLocal()

    log = PredictionLog(
        user_id=1,  # temporary demo user
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

    recommendation = apply_business_rules(
        label,
        data["rainfall"]
    )

    # save audit log
    save_log(label, confidence)

    # confidence color
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