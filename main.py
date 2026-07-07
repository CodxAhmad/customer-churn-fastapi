from fastapi import FastAPI, HTTPException
import joblib

from user_schema import CustomerInput
from user_to_model_level import encode_input

app = FastAPI(title="Customer Churn Prediction")

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is live!", "docs": "/docs", "health": "/health"}

# Load ensemble model
model = joblib.load("models/voting_soft.pkl")

# Load artifacts
scaler = joblib.load("artifacts/scaler.pkl")
model_columns = joblib.load("artifacts/feature_names.pkl")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict")
async def predict(data: CustomerInput):
    try:
        user_dict = data.dict()

        # Encode → align columns
        df = encode_input(user_dict, model_columns)

        # Scale
        X_scaled = scaler.transform(df)

        # Predict
        prob = model.predict_proba(X_scaled)[0][1]
        label = int(prob >= 0.5)
        confidence = prob if label == 1 else 1 - prob
        
        return {
            "prediction": "Customer will churn" if label else "Customer will not churn",
            "probability": round(float(prob), 4),
            "confidence": round(float(confidence), 4),
            "label": label
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
