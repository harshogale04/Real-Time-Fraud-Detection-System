import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))  # add api/ folder itself so 'db' resolves

from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, SCALER_PATH, FRAUD_THRESHOLD
from db import init_db, insert_transaction, get_recent_transactions

app = FastAPI(title="Real-Time Fraud Detection API")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
init_db()


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    actual_label: int = None  # optional, only for simulator/demo accuracy tracking


@app.get("/")
def root():
    return {"status": "Fraud Detection API running"}


@app.post("/predict")
def predict(tx: Transaction):
    data = tx.dict()
    actual_label = data.pop("actual_label", None)

    df = pd.DataFrame([data])
    df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

    # Ensure column order matches training
    feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = df[feature_order]

    prob = float(model.predict_proba(df)[0][1])
    is_flagged = int(prob >= FRAUD_THRESHOLD)

    insert_transaction(
        timestamp=datetime.now().isoformat(),
        amount=data["Amount"],
        fraud_probability=prob,
        is_flagged=is_flagged,
        actual_label=actual_label if actual_label is not None else -1,
    )

    return {
        "fraud_probability": round(prob, 4),
        "is_flagged": bool(is_flagged),
    }


@app.get("/transactions")
def transactions(limit: int = 100):
    rows = get_recent_transactions(limit)
    return [
        {
            "timestamp": r[0],
            "amount": r[1],
            "fraud_probability": r[2],
            "is_flagged": bool(r[3]),
            "actual_label": r[4],
        }
        for r in rows
    ]