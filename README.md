# Real-Time Fraud Detection System

A end-to-end machine learning system that detects fraudulent credit card transactions in real time. Built with a full pipeline covering data preprocessing, class imbalance handling, model training/evaluation, a live prediction API, a transaction stream simulator, and a real-time monitoring dashboard.

## Overview

Credit card fraud is a classic highly-imbalanced classification problem — fraudulent transactions make up less than 0.2% of all transactions. This project builds a system that:

1. Trains ML models to detect fraud despite severe class imbalance
2. Serves predictions through a REST API
3. Simulates a live transaction stream hitting that API
4. Visualizes flagged transactions on a real-time dashboard

## Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | Pandas, NumPy, Scikit-learn |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Models | XGBoost, Random Forest |
| Backend API | FastAPI, Uvicorn |
| Storage | SQLite |
| Dashboard | Streamlit |
| Stream Simulation | Python, Requests |

## Architecture

```
Training Pipeline
──────────────────
┌───────────────────┐     ┌───────────────┐     ┌────────────────┐     ┌───────────┐
│  Kaggle Dataset    │ --> │ Preprocessing │ --> │    Training     │ --> │   Model   │
│ (creditcard.csv)   │     │   + SMOTE     │     │  RF / XGBoost   │     │  (.pkl)   │
└───────────────────┘     └───────────────┘     └────────────────┘     └─────┬─────┘
                                                                              │
                                                                              │ loaded by
                                                                              v
Real-Time Inference Pipeline                                          ┌────────────┐
─────────────────────────────                                        │  FastAPI   │
┌───────────────────┐                                                 │ /predict   │
│ Stream Simulator   │ ─────────────── sends transactions ──────────> │  endpoint  │
│  (sends txns)      │                                                └─────┬──────┘
└───────────────────┘                                                       │
                                                                              v
                                                                       ┌────────────┐
                                                                       │   SQLite   │
                                                                       │ (tx history)│
                                                                       └─────┬──────┘
                                                                              │
                                                                              v
                                                                       ┌────────────┐
                                                                       │ Streamlit  │
                                                                       │ Dashboard  │
                                                                       └────────────┘
```

## Key Features

- **Imbalance-aware training**: Uses SMOTE to oversample the minority (fraud) class instead of relying on misleading accuracy metrics.
- **Correct evaluation metric**: Models are compared using **PR-AUC (Precision-Recall AUC)**, which is far more meaningful than accuracy on a ~0.17% fraud dataset.
- **Model comparison**: Trains both Random Forest and XGBoost, automatically selecting the best performer.
- **Real-time inference**: A FastAPI backend serves live fraud probability scores per transaction.
- **Live monitoring**: A Streamlit dashboard auto-refreshes every 2 seconds, showing flagged transactions, fraud rate, and probability trends.
- **Persistent transaction log**: All predictions are stored in SQLite for auditability.

## Folder Structure

```
fraud-detection-system/
├── data/
│   └── creditcard.csv
├── models/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── test_data.csv
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   ├── main.py
│   ├── db.py
│   └── transactions.db
├── dashboard/
│   └── app.py
├── stream_simulator.py
├── requirements.txt
└── README.md
```

## Setup & Installation

1. **Clone the repo and create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   Get `creditcard.csv` from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in `data/`.

## Running the Project

Run these in order, each in its own terminal:

**1. Train the model**
```bash
cd src
python train.py
```

**2. (Optional) Evaluate the model**
```bash
python evaluate.py
```

**3. Start the API** (from project root)
```bash
uvicorn api.main:app --reload
```

**4. Launch the dashboard** (new terminal, from project root)
```bash
streamlit run dashboard/app.py
```

**5. Start the transaction stream simulator** (new terminal, from project root)
```bash
python stream_simulator.py
```

Watch the dashboard at `http://localhost:8501` update live as transactions stream in.

## Results

- Model selection based on **PR-AUC**, not accuracy, due to extreme class imbalance (~0.17% fraud rate).
- Confusion matrix and precision-recall curve saved automatically after evaluation (`confusion_matrix.png`, `precision_recall_curve.png`).
- Successfully detects rare fraud cases with high confidence in real-time simulation (e.g., correctly flagged fraud with 99.99% predicted probability during live testing).

## Future Improvements

- Replace the script-based stream simulator with Kafka for true event-driven streaming.
- Add model explainability (SHAP values) to justify each fraud flag.
- Deploy the API and dashboard to the cloud (e.g., Render, AWS) for public access.
- Add authentication and rate-limiting to the API.
- Track model drift over time and support periodic retraining.

## Author

Built as an academic project demonstrating applied machine learning, imbalanced classification, backend API design, and real-time data visualization.
