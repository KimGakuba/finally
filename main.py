from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import joblib
import numpy as np
import os
from dotenv import load_dotenv
from database import engine, SessionLocal
from models_db import Base, User, PredictionLog

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

Base.metadata.create_all(bind=engine)

model = joblib.load("smart_irrigation_model.pkl")
encoder = joblib.load("target_encoder.pkl")


@app.route("/")
def home():
    return jsonify({"message": "Smart Irrigation API Running"})


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    db = SessionLocal()
    existing = db.query(User).filter_by(username=data["username"]).first()
    if existing:
        db.close()
        return jsonify({"error": "Username already exists"}), 400
    hashed = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    user = User(username=data["username"], password=hashed)
    db.add(user)
    db.commit()
    db.close()
    return jsonify({"message": "User registered successfully"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    db = SessionLocal()
    user = db.query(User).filter_by(username=data["username"]).first()
    db.close()
    if not user or not bcrypt.check_password_hash(user.password, data["password"]):
        return jsonify({"error": "Invalid credentials"}), 401
    token = create_access_token(identity=str(user.id))
    return jsonify({"token": token})


@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    user_id = int(get_jwt_identity())
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
    recommendation = f"Irrigation level: {label}"

    if confidence > 90:
        color = "green"
    elif confidence > 80:
        color = "yellow"
    else:
        color = "red"

    db = SessionLocal()
    log = PredictionLog(
        user_id=user_id,
        prediction=label,
        confidence=confidence,
        recommendation=recommendation
    )
    db.add(log)
    db.commit()
    db.close()

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "color": color,
        "recommendation": recommendation
    })


@app.route("/history", methods=["GET"])
@jwt_required()
def history():
    user_id = int(get_jwt_identity())
    db = SessionLocal()
    logs = db.query(PredictionLog).filter_by(user_id=user_id).order_by(PredictionLog.timestamp.desc()).all()
    db.close()
    return jsonify([{
        "prediction": l.prediction,
        "confidence": l.confidence,
        "recommendation": l.recommendation,
        "timestamp": l.timestamp.isoformat()
    } for l in logs])


if __name__ == "__main__":
    app.run()
