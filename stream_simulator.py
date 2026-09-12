import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import requests
import random

from config import TEST_DATA_PATH, API_URL

def run_simulation(delay_range=(0.2, 1.5), limit=200):
    df = pd.read_csv(TEST_DATA_PATH)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle

    print(f"Streaming {min(limit, len(df))} transactions to {API_URL}...\n")

    for i, row in df.head(limit).iterrows():
        payload = row.drop("Class").to_dict()
        payload["actual_label"] = int(row["Class"])

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            result = resp.json()
            flag = "🚨 FRAUD" if result["is_flagged"] else "OK"
            print(f"[{i}] Amount: {row['Amount']:.2f} | Prob: {result['fraud_probability']:.4f} | {flag} | Actual: {int(row['Class'])}")
        except Exception as e:
            print(f"Error sending transaction {i}: {e}")

        time.sleep(random.uniform(*delay_range))


if __name__ == "__main__":
    run_simulation()