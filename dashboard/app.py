import streamlit as st
import pandas as pd
import requests
import time

API_URL = "http://127.0.0.1:8000/transactions"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🏦 Real-Time Fraud Detection Dashboard")

placeholder = st.empty()

while True:
    try:
        resp = requests.get(API_URL, params={"limit": 200}, timeout=5)
        data = resp.json()
        df = pd.DataFrame(data)

        with placeholder.container():
            if df.empty:
                st.info("Waiting for transactions... run stream_simulator.py")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", len(df))
                col2.metric("Flagged as Fraud", int(df["is_flagged"].sum()))
                col3.metric("Fraud Rate", f"{df['is_flagged'].mean()*100:.2f}%")

                st.subheader("🚨 Flagged Transactions")
                st.dataframe(
                    df[df["is_flagged"] == True].sort_values("timestamp", ascending=False),
                    use_container_width=True,
                )

                st.subheader("All Recent Transactions")
                st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

                st.subheader("Fraud Probability Over Time")
                st.line_chart(df.set_index("timestamp")["fraud_probability"])

    except Exception as e:
        st.error(f"Could not connect to API: {e}")

    time.sleep(2)