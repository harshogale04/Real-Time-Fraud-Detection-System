import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
TEST_DATA_PATH = os.path.join(MODEL_DIR, "test_data.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# API
API_URL = "http://127.0.0.1:8000/predict"

# DB (used by API + dashboard)
DB_PATH = os.path.join(BASE_DIR, "api", "transactions.db")

# Fraud probability threshold above which a transaction is flagged
FRAUD_THRESHOLD = 0.5