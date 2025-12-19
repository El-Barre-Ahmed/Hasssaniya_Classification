import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_and_vectorizer.pkl")

# Load model & vectorizer
data = joblib.load(MODEL_PATH)
model = data['model']
vectorizer = data['vectorizer']

# FastAPI app
app = FastAPI(
    title="Hassaniya Sentiment Analysis API",
    description="API pour la classification de sentiment des textes en hassaniya",
    version="1.0"
)

# Input schema
class Message(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(msg: Message):
    X = vectorizer.transform([msg.text])
    prediction = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    neg, neu, pos = probs
    polarity_score = float(max(probs))

    return {
        "prediction": prediction,
        "polarity_score": round(polarity_score, 2),
        "probabilities": {
            "positive": round(float(pos), 2),
            "neutral": round(float(neu), 2),
            "negative": round(float(neg), 2)
        }
    }
