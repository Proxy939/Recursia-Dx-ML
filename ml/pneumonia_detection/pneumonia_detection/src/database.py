"""
database.py — SQLite prediction history.
"""

import sqlite3
import os
from datetime import datetime
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "predictions.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            filename    TEXT,
            label       TEXT,
            confidence  REAL,
            prob_pneumonia REAL,
            dn_prob     REAL,
            en_prob     REAL,
            severity    TEXT,
            affected_pct REAL,
            threshold   REAL
        )
    """)
    conn.commit()
    return conn


def save_prediction(filename: str, result: dict, severity: dict, threshold: float):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO predictions
        (timestamp, filename, label, confidence, prob_pneumonia,
         dn_prob, en_prob, severity, affected_pct, threshold)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename,
        result["label"],
        result["confidence"],
        result["prob_pneumonia"],
        result["dn_prob"],
        result["en_prob"],
        severity.get("severity", "N/A"),
        severity.get("affected_pct", 0.0),
        threshold
    ))
    conn.commit()
    conn.close()


def get_history() -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df


def clear_history():
    conn = _get_conn()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
