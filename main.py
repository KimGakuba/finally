from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("smart_irrigation_model.pkl")
encoder = joblib.load("target_encoder.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Smart Irrigation API Running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [[
        data["temperature"],
        data["humidity"],
        data["soil_moisture"],
        data["rainfall"]
    ]]
    pred = model.predict(features)
    probs = model.predict_proba(features)
    confidence = round(float(np.max(probs) * 100), 2)
    label = encoder.inverse_transform(pred)[0]

    if confidence > 90:
        color = "green"
    elif confidence > 80:
        color = "yellow"
    else:
        color = "red"

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "color": color,
        "recommendation": f"Irrigation level: {label}"
    })

if __name__ == "__main__":
    app.run()
