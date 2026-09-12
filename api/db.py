import sqlite3
from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            fraud_probability REAL,
            is_flagged INTEGER,
            actual_label INTEGER
        )
    """)
    conn.commit()
    conn.close()


def insert_transaction(timestamp, amount, fraud_probability, is_flagged, actual_label):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO transactions (timestamp, amount, fraud_probability, is_flagged, actual_label) VALUES (?, ?, ?, ?, ?)",
        (timestamp, amount, fraud_probability, is_flagged, actual_label),
    )
    conn.commit()
    conn.close()


def get_recent_transactions(limit=100):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, amount, fraud_probability, is_flagged, actual_label FROM transactions ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows