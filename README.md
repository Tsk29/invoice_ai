# Invoice AI

A multilingual invoice parsing application with fraud detection, Isolation Forest-based
anomaly detection, and interactive data visualization — built on Groq's vision LLMs.

*Process invoices from images or PDFs in seconds ⏱️*


##  Features

- **Multilingual Support**: Parse invoices in English, Spanish, French, German, Tamil, and more
- **AI-Powered Extraction**: Uses a Groq-hosted vision LLM for structured data extraction from images or PDFs
- **PDF Support**: Upload PDF invoices directly — pages are rasterized and sent through the same extraction pipeline as images
- **Fraud Detection**: Rules-based system flagging duplicate invoice numbers, unusually high totals, and disproportionate tax
- **Anomaly Detection**: Real Isolation Forest (scikit-learn) over total/subtotal/tax to flag statistically unusual invoices
- **Interactive UI**: Streamlit dashboard with a working light/dark mode toggle
- **Batch Processing**: Upload and process multiple invoices (images or PDFs) at once
- **Data Export**: Export extracted invoices to CSV or JSON with one click
- **Chat Assistant**: Ask natural-language questions about your processed invoices




##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-00A98F?logo=groq)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF_Rendering-red)
![Isolation Forest](https://img.shields.io/badge/Isolation_Forest-scikit--learn-green?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly)

> **Note on the model**: the extraction model is a config value (`GroqClient`, `utils.py`), not
> hardcoded to one vendor's release. Groq's catalog of available vision-capable models changes
> over time — check `GET /openai/v1/models` on your own API key and update the default in
> `utils.py` if the configured model stops being served.


### Key Directories Explained:

1. **`.Dataset/`**  
   ![Dataset Icon](https://img.icons8.com/color/48/000000/database.png)  
   Curated collection of sample invoices in multiple languages and formats for testing and development.

2. **`.streamlit/`**  
   ![Config Icon](https://img.icons8.com/color/48/000000/settings.png)  
   Contains environment configurations including:
   - API keys (secured via `secrets.toml`)
   - UI theme settings
   - Performance configurations

3. **`Results/`**  
   ![Results Icon](https://img.icons8.com/color/48/000000/data-configuration.png)  
   Organized outputs including:
   - Structured JSON/CSV exports
   - Fraud detection reports
   - Interactive visualizations

4. **Core Modules**  
   ![Python Icon](https://img.icons8.com/color/48/000000/python.png)  
   - `analytics.py`: ML-powered anomaly detection
   - `app.py`: Main processing pipeline
   - `enhanced_ui.py`: Interactive dashboard components

##  How It Works

### 1. Intelligent Invoice Parsing Pipeline

![image](https://github.com/user-attachments/assets/2b213cc7-2ecc-403c-a979-d72e846aefbc)

Our system follows a sophisticated multi-stage process:

1. **PDF Rasterization**: PDF uploads are rendered page-by-page to images (PyMuPDF) before extraction
2. **Image Preprocessing**: Enhances contrast and resizes images for optimal OCR accuracy
3. **Vision LLM Extraction**: A Groq-hosted vision model extracts structured data from the image
4. **Type Detection**: Classifies invoices as retail, service, utility, or general
5. **Confidence Scoring**: Provides confidence levels for each extracted field
6. **Validation**: Cross-checks calculated totals against extracted values

### 2. Anomaly Detection with Isolation Forest

`analytics.py` runs a real scikit-learn Isolation Forest over each batch's numeric fields
(`total_amount`, `subtotal`, `tax`), surfaced in the **Analytics** tab of the dashboard
(`enhanced_ui.py`) once at least 4 invoices have been processed:

```python
from sklearn.ensemble import IsolationForest

# Prepare features for anomaly detection
features = df[['total_amount', 'tax', 'subtotal']].dropna(how='all')

# Train Isolation Forest model
clf = IsolationForest(n_estimators=200, contamination=0.15, random_state=42)
predictions = clf.fit_predict(features)

# Predict anomalies
df['anomaly_score'] = clf.decision_function(features)
df['is_anomaly'] = predictions == -1
```

Key advantages:
- Effectively handles high-dimensional data
- No need for labeled anomaly data
- Identifies both global and local outliers
- Computationally efficient

### 3. Fraud Detection System

`detect_fraud()` (`enhanced_ui.py`) applies rule-based checks per batch:

- Duplicate invoice numbers across different uploads
- Unusually high total amount (> 100,000)
- Tax disproportionate to total (> 30% of total amount)

This runs independently from the Isolation Forest anomaly detection in the **Analytics**
tab — fraud rules are deterministic thresholds, anomaly detection is statistical.

##  UI Showcase

| Feature | Screenshot |
|---------|------------|
| **Main Interface** | ![image](https://github.com/user-attachments/assets/2e8c9582-ff0c-4790-817f-95298c7d43f2)|
| **Fraud Detection** |![Screenshot 2025-06-03 231430](https://github.com/user-attachments/assets/c9531a28-631a-46b8-9574-ae2e7d02c00f)|
| **Chat Assistant** |![image](https://github.com/user-attachments/assets/11de9800-101e-4099-bf3c-b81815efbc83)|

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tsk29/invoice_ai.git
   cd invoice_ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Groq API key (get one at [console.groq.com](https://console.groq.com)) using **either**:
   - a `.streamlit/secrets.toml` file:
     ```toml
     GROQ_API_KEY = "your_api_key_here"
     ```
   - or a `.env` file (used when running with `--environment local`):
     ```
     GROQ_API_KEY=your_api_key_here
     ```

4. Run the application. There are two entry points:
   ```bash
   # Full dashboard: batch upload, fraud detection, analytics, chatbot tabs
   streamlit run enhanced_ui.py

   # Lightweight single-invoice extractor
   streamlit run app.py
   ```
   Add `-- --environment local` to either command to read the key from `.env` instead of
   `secrets.toml`.

##  Performance Metrics

There's no automated benchmark harness in this repo yet, so no accuracy/precision/recall
numbers are published here — publishing unmeasured figures would be misleading. Adding a
benchmark harness (planted-defect scoring against ground truth, similar in spirit to
extraction-verification setups) is a good candidate for a future enhancement.

##  Chatbot Examples

**User**: "Which invoice has the highest total?"  
**Bot**: "Invoice #INV-7892 has the highest total of $12,450.00 dated 2023-11-15 from VendorTech Solutions."

**User**: "Are there any duplicate invoice numbers?"  
**Bot**: "Yes, invoice number INV-5421 appears 3 times from different vendors. This might indicate fraud."

##  Advanced Analytics

The **Analytics** tab (`analyze_invoices()` in `analytics.py`) charts the current batch with Plotly:

- Distribution of total amounts (histogram)
- Tax vs. subtotal (scatter)
- Invoices by currency (pie chart)
- Isolation Forest anomaly flags (see above)

Temporal trend analysis, vendor spend rollups, and cash-flow forecasting aren't implemented
yet — see [Future Enhancements](#-future-enhancements).

##  Multilingual Support

The application seamlessly handles invoices in multiple languages:

| Language | Sample Output |
|----------|---------------|
| Tamil | ![image](https://github.com/user-attachments/assets/90c73647-7f96-4605-98c1-60cf98fb1a3f)|
| French | ![image](https://github.com/user-attachments/assets/805aafeb-2d7b-4766-bb5a-c02f20c55db8)|

##  Fraud Detection Rules

Implemented today, in `detect_fraud()`:

1. **Duplicate Invoice Numbers**: Same invoice number appears more than once in a batch
2. **Unusually High Total**: Total amount exceeds 100,000
3. **Amount Discrepancies**: Tax exceeds 30% of the total amount

Round-amount detection, after-hours invoice flags, and rapid-succession-from-same-vendor
checks are not implemented yet — tracked in [Future Enhancements](#-future-enhancements).

##  Future Enhancements

- [ ] **Persistence** — invoices currently live only in `st.session_state` and vanish on
      reload; add a SQLite/Postgres store so extracted history survives across sessions
- [ ] **Multi-page PDF extraction** — currently one page per PDF is extracted; merge line
      items across all pages of a multi-page invoice
- [ ] **OCR fallback** — fall back to Tesseract for low-confidence or low-quality scans
      instead of relying solely on the vision LLM
- [ ] Additional fraud rules: round-amount detection, after-hours invoice dates,
      rapid-succession invoices from the same vendor
- [ ] Vendor reputation scoring system
- [ ] Approval workflow / status tracking (pending → approved → paid)
- [ ] Multi-currency conversion and rollup reporting
- [ ] Email ingestion (forward invoices to a monitored inbox for auto-processing)
- [ ] Predictive analytics for payment delays
- [ ] Mobile app with camera integration

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Project Link: https://github.com/Tsk29/invoice_ai

---

✨ **Transform your invoice processing from chore to strategic advantage with AI-powered insights!** ✨
