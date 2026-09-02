import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageChops
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

def compute_anomaly_flags(invoices, contamination: float = 0.15) -> dict:
    """Run Isolation Forest over the batch's numeric fields and return
    {image_id: True/False} for whether each invoice was flagged. Shared by
    the Analytics tab display and the fraud scorer so both agree on the same
    run rather than fitting two separate models on the same data."""
    if len(invoices) < 4:
        return {}

    df = pd.DataFrame([inv["invoice"].dict() for inv in invoices])
    df["image_id"] = [inv["image_id"] for inv in invoices]

    feature_cols = [c for c in ("total_amount", "subtotal", "tax") if c in df]
    if not feature_cols:
        return {}

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid_rows = features.dropna(how="all")
    if len(valid_rows) < 4:
        return {}

    valid_rows = valid_rows.fillna(valid_rows.median(numeric_only=True)).fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=min(contamination, 0.5),
        random_state=42,
    )
    predictions = model.fit_predict(valid_rows)

    flags = {img_id: False for img_id in df["image_id"]}
    for idx, is_anomaly in zip(valid_rows.index, predictions == -1):
        flags[df.loc[idx, "image_id"]] = bool(is_anomaly)
    return flags

def detect_anomalies(invoices, contamination: float = 0.15):
    """Display an Isolation Forest anomaly table for the current batch. Falls
    back to a note when there isn't enough data for the model to fit
    meaningfully."""
    st.subheader("Anomaly Detection (Isolation Forest)")
    feature_cols = [c for c in ("total_amount", "subtotal", "tax") if c in InvoiceData.model_fields]

    if len(invoices) < 4:
        st.info("Not enough invoices yet for Isolation Forest to detect meaningful anomalies (need at least 4).")
        return

    flags = compute_anomaly_flags(invoices, contamination=contamination)
    if not flags:
        st.info("No numeric fields available for anomaly detection.")
        return

    df = pd.DataFrame([inv["invoice"].dict() for inv in invoices])
    df["image_id"] = [inv["image_id"] for inv in invoices]
    df["is_anomaly"] = df["image_id"].map(flags)
    anomalies = df[df["is_anomaly"] == True]

    display_cols = [c for c in ("invoice_number", "vendor_name", *feature_cols, "currency") if c in anomalies]
    if not anomalies.empty:
        st.warning(f"⚠️ Isolation Forest flagged {len(anomalies)} potential anomaly/anomalies:")
        st.dataframe(anomalies[display_cols])
    else:
        st.success("✅ No anomalies detected by Isolation Forest.")

# ---------------------------
# Explainable Fraud Scoring
# ---------------------------

def compute_ela_score(image_bytes: bytes, quality: int = 90) -> float | None:
    """Error Level Analysis: re-save the image at a known JPEG quality and
    measure how much it differs from the original. Genuine, uniformly
    compressed photos settle to a low, even error level; a region that was
    pasted in or re-edited after the original compression tends to show a
    different error level than the rest of the image, which raises the
    overall diff spread. This is a heuristic signal used in real digital
    forensics workflows, not proof of tampering - a clean, unedited photo can
    still score above threshold, and a competent edit can evade it. Returns
    the standard deviation of the pixel-wise difference map, or None if the
    image couldn't be decoded.
    """
    try:
        original = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    buffer = BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    diff = ImageChops.difference(original, resaved)
    diff_array = np.asarray(diff, dtype=np.float32)
    return float(diff_array.std())

# Common invoice date formats. If a date string matches none of these, it's
# flagged as an inconsistent/unparseable format rather than silently ignored -
# a genuinely malformed date is itself a mild red flag on a document that's
# supposed to follow a standard layout.
_DATE_FORMATS = (
    "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
    "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y",
    "%m/%d/%y", "%d/%m/%y",
)

def _is_parseable_date(value: str) -> bool:
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(value.strip(), fmt)
            return True
        except ValueError:
            continue
    return False

def _is_suspiciously_round(amount: float) -> bool:
    """Flags totals like 1000.00, 5000.00, 10000.00 - real transaction totals
    landing exactly on a round thousand are statistically less common than
    ones with cents/odd amounts, a heuristic related to Benford's-law-style
    checks used in real fraud triage. Deliberately conservative (>=1000 and
    an exact multiple of 1000) to avoid flagging ordinary small round bills
    like a $20.00 lunch receipt."""
    return amount >= 1000 and amount % 1000 == 0

REQUIRED_FIELDS = ("invoice_number", "vendor_name", "total_amount", "invoice_date")

# Weight of each triggered signal toward the 0-100 fraud score, and the
# score thresholds that bucket the total into a risk category. These are
# hand-set heuristic weights (not fit on labeled fraud data - there isn't
# any in this repo) - tune them if you have real examples to calibrate against.
FRAUD_SIGNAL_WEIGHTS = {
    "duplicate_invoice_number": 35,
    "unusually_high_total": 20,
    "disproportionate_tax": 20,
    "arithmetic_inconsistency": 15,
    "low_extraction_confidence": 15,
    "statistical_amount_anomaly": 25,
    "possible_image_manipulation": 20,
    "missing_required_fields": 20,
    "inconsistent_date_format": 10,
    "suspicious_value": 10,
}
# Calibrated against .Dataset samples: native-JPEG photos scored 0.5-0.8,
# PNG-sourced/rescanned invoices scored 4.5-5.3 just from format conversion
# noise, with no tampering involved. The threshold sits well above that
# clean baseline to avoid flagging ordinary format differences, which also
# means it will miss anything but fairly aggressive edits - see the
# docstring above and the README for this technique's real limits.
ELA_MANIPULATION_THRESHOLD = 15.0
RISK_THRESHOLDS = (30, 60)  # < low, low-medium boundary, medium-high boundary

def score_invoices(invoices, run_ela: bool = True) -> list[dict]:
    """Score every invoice in the batch for fraud risk. Returns a list of
    dicts (same order as `invoices`) shaped as:
        {"image_id": ..., "risk_level": "Low"/"Medium"/"High",
         "risk_score": 0.0-1.0, "reasons": [...]}
    Every reason maps to something this function actually computed - nothing
    here is a label without a check behind it. In particular, the image-
    manipulation reason flags the *whole image* as having an uneven
    compression pattern (see compute_ela_score) - it does not localize which
    region was altered, so its wording deliberately doesn't claim that."""
    invoice_numbers = [inv["invoice"].invoice_number for inv in invoices if inv["invoice"].invoice_number]
    duplicate_numbers = {num for num in invoice_numbers if invoice_numbers.count(num) > 1}
    anomaly_flags = compute_anomaly_flags(invoices)

    results = []
    for inv in invoices:
        invoice = inv["invoice"]
        confidence_scores = inv.get("confidence_scores") or {}
        reasons = []
        score = 0

        missing_fields = [f for f in REQUIRED_FIELDS if getattr(invoice, f) is None]
        if missing_fields:
            reasons.append(f"Missing required field(s): {', '.join(missing_fields)}.")
            score += FRAUD_SIGNAL_WEIGHTS["missing_required_fields"]

        if invoice.invoice_date and not _is_parseable_date(invoice.invoice_date):
            reasons.append(f"Inconsistent or unrecognized date format: '{invoice.invoice_date}'.")
            score += FRAUD_SIGNAL_WEIGHTS["inconsistent_date_format"]

        if invoice.total_amount is not None and invoice.total_amount <= 0:
            reasons.append(f"Suspicious value: total amount is zero or negative ({invoice.total_amount:,.2f}).")
            score += FRAUD_SIGNAL_WEIGHTS["suspicious_value"]
        elif invoice.total_amount is not None and _is_suspiciously_round(invoice.total_amount):
            reasons.append(f"Suspicious value: total amount is an unusually round figure ({invoice.total_amount:,.2f}).")
            score += FRAUD_SIGNAL_WEIGHTS["suspicious_value"]

        if invoice.invoice_number in duplicate_numbers:
            reasons.append("Duplicate pattern detected: this invoice number appears more than once in the batch.")
            score += FRAUD_SIGNAL_WEIGHTS["duplicate_invoice_number"]

        if invoice.total_amount and invoice.total_amount > 100000:
            reasons.append(f"Unusually high total amount ({invoice.total_amount:,.2f}).")
            score += FRAUD_SIGNAL_WEIGHTS["unusually_high_total"]

        if invoice.tax and invoice.total_amount and invoice.tax > 0.3 * invoice.total_amount:
            reasons.append("Disproportionate tax: tax exceeds 30% of the total amount.")
            score += FRAUD_SIGNAL_WEIGHTS["disproportionate_tax"]

        if invoice.subtotal is not None and invoice.tax is not None and invoice.total_amount is not None:
            expected_total = invoice.subtotal + invoice.tax
            tolerance = max(1.0, 0.02 * invoice.total_amount)
            if abs(expected_total - invoice.total_amount) > tolerance:
                reasons.append(
                    f"OCR inconsistency: subtotal + tax ({expected_total:,.2f}) does not match "
                    f"the extracted total ({invoice.total_amount:,.2f})."
                )
                score += FRAUD_SIGNAL_WEIGHTS["arithmetic_inconsistency"]

        numeric_confidences = [v for v in confidence_scores.values() if isinstance(v, (int, float))]
        if numeric_confidences and (sum(numeric_confidences) / len(numeric_confidences)) < 0.6:
            reasons.append("OCR inconsistency: average field-extraction confidence is below 60%.")
            score += FRAUD_SIGNAL_WEIGHTS["low_extraction_confidence"]

        if anomaly_flags.get(inv["image_id"]):
            reasons.append("Statistical anomaly: this invoice's amounts are an outlier versus the rest of the batch (Isolation Forest).")
            score += FRAUD_SIGNAL_WEIGHTS["statistical_amount_anomaly"]

        if run_ela and inv.get("image_bytes"):
            ela_score = compute_ela_score(inv["image_bytes"])
            if ela_score is not None and ela_score > ELA_MANIPULATION_THRESHOLD:
                reasons.append(
                    f"Possible image manipulation: Error Level Analysis found an uneven compression "
                    f"pattern across the image (score {ela_score:.1f}, threshold {ELA_MANIPULATION_THRESHOLD:.0f}) - "
                    f"a whole-image heuristic signal, not proof of editing and not localized to a specific region."
                )
                score += FRAUD_SIGNAL_WEIGHTS["possible_image_manipulation"]

        score = min(score, 100)
        low_bound, high_bound = RISK_THRESHOLDS
        risk_level = "Low" if score < low_bound else ("Medium" if score < high_bound else "High")

        results.append({
            "image_id": inv["image_id"],
            "risk_level": risk_level,
            "risk_score": round(score / 100, 2),
            "reasons": reasons,
        })
    return results