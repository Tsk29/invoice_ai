import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
from utils import InvoiceData

def analyze_invoices(invoices):
    """Display an analytics dashboard for processed invoices."""
    df = pd.DataFrame([inv["invoice"].dict() for inv in invoices])
    
    # Total Amount Distribution
    if "total_amount" in df and df["total_amount"].notna().any():
        st.subheader("Total Amount Distribution")
        fig = px.histogram(df, x="total_amount", nbins=20, title="Distribution of Total Amounts")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tax vs. Subtotal Scatter
    if "subtotal" in df and "tax" in df and df[["subtotal", "tax"]].notna().any().all():
        st.subheader("Tax vs. Subtotal")
        fig = px.scatter(df, x="subtotal", y="tax", title="Tax vs. Subtotal", hover_data=["invoice_number"])
        st.plotly_chart(fig, use_container_width=True)
    
    # Currency Breakdown
    if "currency" in df and df["currency"].notna().any():
        st.subheader("Currency Breakdown")
        currency_counts = df["currency"].value_counts().reset_index()
        currency_counts.columns = ["Currency", "Count"]
        fig = px.pie(currency_counts, names="Currency", values="Count", title="Invoices by Currency")
        st.plotly_chart(fig, use_container_width=True)

def detect_anomalies(invoices, contamination: float = 0.15):
    """Detect anomalous invoices using an Isolation Forest over numeric fields
    (total_amount, subtotal, tax). Falls back to a note when there isn't enough
    data for the model to fit meaningfully."""
    st.subheader("Anomaly Detection (Isolation Forest)")
    df = pd.DataFrame([inv["invoice"].dict() for inv in invoices])

    feature_cols = [c for c in ("total_amount", "subtotal", "tax") if c in df]
    if not feature_cols:
        st.info("No numeric fields available for anomaly detection.")
        return

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid_rows = features.dropna(how="all")
    if len(valid_rows) < 4:
        st.info("Not enough invoices yet for Isolation Forest to detect meaningful anomalies (need at least 4).")
        return

    valid_rows = valid_rows.fillna(valid_rows.median(numeric_only=True)).fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=min(contamination, 0.5),
        random_state=42,
    )
    predictions = model.fit_predict(valid_rows)
    scores = model.decision_function(valid_rows)

    result = df.loc[valid_rows.index].copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = predictions == -1
    anomalies = result[result["is_anomaly"]].sort_values("anomaly_score")

    display_cols = [c for c in ("invoice_number", "vendor_name", *feature_cols, "currency", "anomaly_score") if c in anomalies]
    if not anomalies.empty:
        st.warning(f"⚠️ Isolation Forest flagged {len(anomalies)} potential anomaly/anomalies:")
        st.dataframe(anomalies[display_cols])
    else:
        st.success("✅ No anomalies detected by Isolation Forest.")