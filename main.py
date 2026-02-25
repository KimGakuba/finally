from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import gdown
import os

app = Flask(__name__)
CORS(app)

# Download models if not present
MODEL_PATH = "smart_irrigation_model.pkl"
ENCODER_PATH = "target_encoder.pkl"

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=1MvqPRk7KnfO9GNVpftKOCvtsyLO8AnHQ",
        MODEL_PATH,
        quiet=False,
        fuzzy=True
    )

if not os.path.exists(ENCODER_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=13dCZ-773wciiFKQjua-Q2dcoHNnfScZt",
        ENCODER_PATH,
        quiet=False,
        fuzzy=True
    )

# Load model
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

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
```

**`requirements.txt`**
```
flask==3.0.3
flask-cors==4.0.1
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
gdown==5.2.0
gunicorn==22.0.0
```

**Render start command:**
```
gunicorn main:app --bind 0.0.0.0:$PORT
