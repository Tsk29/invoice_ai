# Invoice AI

A multilingual invoice parsing application with fraud detection, Isolation Forest-based
anomaly detection, and interactive data visualization — built on Groq's vision LLMs.

*Process invoices from images or PDFs in seconds ⏱️*


##  Features

- **Multilingual Support**: Parse invoices in English, Spanish, French, German, Tamil, and more
- **AI-Powered Extraction**: Uses a Groq-hosted vision LLM for structured data extraction from images or PDFs, with an optional PaddleOCR + text-LLM hybrid path (see [OCR Architecture](#ocr-architecture))
- **PDF Support**: Upload PDF invoices directly — pages are rasterized and sent through the same extraction pipeline as images
- **Explainable Fraud Scoring**: Every invoice gets a 0-100 score, a Low/Medium/High risk category, and a plain-language reason for every signal that fired - not just a flag with no explanation
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
![Pillow](https://img.shields.io/badge/Pillow-ELA_%2F_Preprocessing-blueviolet)
![Isolation Forest](https://img.shields.io/badge/Isolation_Forest-scikit--learn-green?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly)

> **Note on the model**: this currently runs on `qwen/qwen3.8-27b` (Alibaba's Qwen), not Llama -
> the project originally targeted Meta's `llama-4-scout` vision model, but Groq stopped serving
> it, and as of this writing Groq has **no active Llama vision model at all**; Qwen's two
> vision-capable models are the only ones currently available there. The model is a config
> value (`GroqClient`, `utils.py`), not hardcoded to one vendor - check `GET /openai/v1/models`
> on your own API key and update the default in `utils.py` if the configured model stops being
> served, which is exactly what happened here once already.


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
   - `analytics.py`: Isolation Forest anomaly detection, explainable fraud scoring, ELA
     image-manipulation heuristic, Plotly charts
   - `app.py`: Lightweight single-invoice processing pipeline
   - `enhanced_ui.py`: Full dashboard - batch extraction, chatbot, fraud detection, analytics

##  Interface Tour

`enhanced_ui.py` is the primary app — a single-page dashboard with a sidebar and four tabs.
(There's no hosted screenshot here; the UI changes too often for a static image to stay
accurate, so here's what's actually on screen.)

**Sidebar** — invoice language picker (English, Spanish, French, German, Tamil, Other), a
🌙 dark-mode toggle (dark by default) that re-themes the whole page, live batch status as
metric tiles (total, processed, success rate), and a "Clear all invoices" button that resets
the session (invoices and chat history) once there's something to clear.

**📄 Invoice Extraction** — drag-and-drop upload for one or many images/PDFs at once;
multi-page PDFs get a page picker. A radio picks the extraction method: the default vision LLM,
or (if `requirements-ocr.txt` is installed) PaddleOCR + Text LLM - see
[OCR Architecture](#ocr-architecture). Each processed image renders as a bordered card: a
metric row (invoice #, vendor, total, date), a line-items table, per-image type/confidence/
timing captions, a collapsible raw-confidence-scores view, and a collapsible full-JSON view for
anyone who wants the unformatted data - plus direct "Download All as CSV / JSON" buttons and an
editable data grid (low-confidence cells highlighted) for the whole batch.

**🤖 Chatbot** — a real chat thread (`st.chat_message` bubbles) with four one-click quick
questions plus free-text input, answering from the currently extracted invoice batch (capped at
the 15 most recent invoices per question, so a large batch doesn't blow up prompt size - the
model is told explicitly when it's only seeing a subset). Both the quick-question buttons and
the input are disabled with an explanatory message until at least one invoice exists, so asking
a question never fires against empty context. A "Clear chat" button appears once there's a
conversation to clear.

**🚨 Fraud Detection** — one expandable card per invoice: Vendor/Invoice Number/Date/Amount as
metrics, a risk score progress bar, a Low/Medium/High risk level, a human-readable explanation
list, and a "Generated assessment (JSON)" popover with the exact scoring output (see
[Fraud Scoring](#fraud-scoring) below).

**📊 Analytics** — Plotly charts (total-amount distribution, tax vs. subtotal, currency
breakdown) plus the Isolation Forest anomaly table.

`app.py` is a second, lightweight entry point: single-invoice upload, extraction, inline
editing, CSV/JSON export, and a basic chat box — no tabs, no batch processing.

##  How It Works

### Extraction Pipeline

The extraction flow (used by both apps) is:

1. **PDF Rasterization**: PDF uploads are rendered page-by-page to images (PyMuPDF) before extraction
2. **Image Preprocessing**: Enhances contrast and resizes images for optimal OCR accuracy
3. **Vision LLM Extraction**: A Groq-hosted vision model extracts structured data from the image
4. **Type Detection**: Classifies invoices as retail, service, utility, or general - computed
   from the same extraction call's result, not a separate model call
5. **Confidence Scoring**: Provides confidence levels for each extracted field
6. **Validation**: Cross-checks calculated totals against extracted values

**Batch extraction speed**: `enhanced_ui.py`'s batch uploader used to make *two* full vision-LLM
calls per image - one solely to classify invoice type (discarding everything except that
label), then a second, nearly identical call that was the one actually kept. Type detection
only needs fields the extraction call already returns (vendor name, line-item descriptions),
so the second call was pure overhead. It's now one call per image, and the batch runs
concurrently (`ThreadPoolExecutor`, capped at 4 workers) instead of one image at a time.
Measured on a 4-image batch against the same model: ~230s before → ~140s after (~40% faster).
The remaining time comes from the vision model's own per-call latency, which - unlike the
optional PaddleOCR path - varies with Groq's current load rather than being a fixed cost. See
[OCR Architecture](#ocr-architecture) below for the two paths compared head-to-head.

### Anomaly Detection with Isolation Forest

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

##  Fraud Scoring

`score_invoices()` (`analytics.py`) scores every invoice in a batch internally on a 0-100
scale, then returns a normalized result per invoice shaped exactly as:

```json
{
  "risk_level": "High",
  "risk_score": 0.89,
  "reasons": [
    "Duplicate pattern detected: this invoice number appears more than once in the batch.",
    "Possible image manipulation: Error Level Analysis found an uneven compression pattern across the image (score 22.4, threshold 15) - a whole-image heuristic signal, not proof of editing and not localized to a specific region."
  ]
}
```

`risk_level` buckets from the internal 0-100 score as **Low** (<30) / **Medium** (30-59) /
**High** (60+); `risk_score` is that score divided by 100, rounded to 2 decimals. Every string
in `reasons` maps to a check the function actually ran — nothing here is a label without a
check behind it:

| Signal | Weight | What it checks |
|---|---|---|
| Duplicate pattern detected | 35 | Invoice number appears more than once in the batch |
| Unusually high total | 20 | `total_amount` > 100,000 |
| Disproportionate tax | 20 | `tax` > 30% of `total_amount` |
| OCR inconsistency (arithmetic) | 15 | `subtotal + tax` doesn't match the extracted `total_amount` |
| OCR inconsistency (confidence) | 15 | Average field-extraction confidence < 60% |
| Statistical anomaly | 25 | Isolation Forest flags this invoice's amounts as an outlier vs. the rest of the batch (shares one model fit with the Analytics tab, so both agree) |
| Missing required field(s) | 20 | Any of `invoice_number`, `vendor_name`, `total_amount`, `invoice_date` is null |
| Inconsistent date format | 10 | `invoice_date` doesn't match any of ~10 common date patterns tried |
| Suspicious value | 10 | `total_amount` is zero/negative, or an exact multiple of 1,000 (a mild Benford's-law-style heuristic - real totals landing on an exact round thousand are less common than one with cents) |
| Image manipulation | 20 | Error Level Analysis (ELA) finds an uneven JPEG compression pattern across the whole image - **not** localized to a region, see caveat below |

Weights sum and cap at 100 before the /100 normalization. These are hand-set heuristic
weights, not fit on labeled fraud data — there isn't any in this repo — so treat the score as
a triage signal, not a verdict. The **Fraud Detection** tab shows this exact JSON per invoice
behind a "Generated assessment (JSON)" button, alongside the extracted Vendor / Invoice
Number / Date / Amount and a human-readable explanation list.

**On the image-manipulation signal specifically**: ELA re-saves the image at a known JPEG
quality and measures how much it differs from the original; a region edited after the
original compression tends to show a different error level than the rest of the image. It's
a real, if weak, forensics heuristic — calibrated here against `.Dataset` samples, where
native-JPEG photos scored 0.5-0.8 and PNG-sourced/rescanned invoices scored 4.5-5.3 from
format-conversion noise alone, with no tampering involved. The threshold (15.0) sits above
that clean baseline to avoid flagging ordinary format differences, which also means it will
miss anything but fairly aggressive edits. A synthetic tamper test (pasting text into a
sample invoice) barely moved the score. Treat this as a weak supplementary signal, not a
manipulation detector.

##  OCR Architecture

By default, the extraction pipeline sends the whole invoice image straight to a Groq-hosted
vision LLM, which reads and structures it in one pass. `ocr_hybrid.py` adds an **optional**
second path: PaddleOCR reads the raw text locally, then a text-only LLM call (no image tokens)
structures that text into the same schema. Both are selectable per-batch from a radio in the
Extraction tab, and the choice is a genuine engineering trade-off, not a strict upgrade in
either direction - here's what an actual spike found, run against `.Dataset/` samples on a
CPU-only dev machine (no GPU):

| | Vision LLM (default) | PaddleOCR + Text LLM |
|---|---|---|
| Single-call latency, English invoice (7 line items) | ~2.7s | ~24.4s (~22s OCR + ~2.4s LLM) |
| Single-call latency, Tamil invoice (3 line items) | ~2.0s | ~11.8s (~10s OCR + ~1.7s LLM) |
| Accuracy, English invoice | Correct - matched hybrid exactly | Correct |
| Accuracy, Tamil invoice | Missed `tax` entirely on this run | Correct, including `tax` |
| Dependencies | None beyond the base app | `paddlepaddle` (~100MB macOS ARM / ~186MB Linux) + `paddleocr` + ~250-400MB of downloaded recognition/detection models |
| Setup | Works out of the box | Needs `requirements-ocr.txt` installed in a **virtual environment** (see below) |

**Verdict**: Vision LLM is the better default. On a single call it was consistently 5-9x faster
than the hybrid path in both languages tested, and matched it exactly on the English sample.
Earlier notes here claimed vision latency "swings wildly, 2s-30s+" - that variability shows up
under concurrent *batch* load (multiple images submitted together, likely Groq queuing/rate
limits), not on a single image, which was fast and consistent in both runs above. The one place
the hybrid path earned its cost was Tamil `tax` extraction, which lines up with a mechanical
reason: PaddleOCR isolates a line like `TOTAL` / `Rs.1,370.00` as its own clean text region,
giving the structuring LLM less room to reinterpret or recompute it than reading the number off
a full table layout in an image. Worth reaching for on a non-Latin-script invoice you don't
trust the default extraction on - not a general replacement.

### How the hybrid path works, step by step

1. The extraction-method radio in the UI passes `method="paddleocr_hybrid"` into
   `extract_one_invoice()` (`enhanced_ui.py`).
2. `ocr_hybrid.extract_text_paddleocr()` writes the image bytes to a temp file - PaddleOCR's
   `predict()` only accepts a file path, not bytes - and runs PaddleOCR's own detection →
   recognition pipeline on it, returning a flat string of recognized text lines.
3. That text (never the image) is substituted into `STRUCTURING_PROMPT_TEMPLATE` and sent via
   `GroqClient.structure_text()` - a plain-string chat message with no `image_url` content
   block, which is why it's cheaper and faster per call than the vision path.
4. The response is parsed into the same `InvoiceData` schema and merged into the same record
   shape (`image_id`, `confidence_scores`, `invoice_type`, timing) as the vision path, so
   everything downstream - fraud scoring, analytics, export - treats both paths identically.
5. If `paddleocr` isn't installed, `ocr_hybrid.py`'s import is wrapped in try/except and sets
   `PADDLEOCR_AVAILABLE = False`; the UI checks that flag and simply doesn't render the option.

### Pros and cons

**PaddleOCR + Text LLM**
- ✅ Caught a `tax` field the vision model missed on the Tamil sample (small-sample signal, not proven)
- ✅ Text-only LLM call is cheaper per token than a vision call (no image tokens)
- ✅ Fully local OCR step - no image data leaves the machine until the (much smaller) text payload is sent
- ❌ 5-9x slower end-to-end on both samples tested here
- ❌ ~350-600MB of dependencies (paddlepaddle + paddleocr + downloaded models), needs its own virtual environment
- ❌ Loses layout/table structure - OCR returns a flat list of text lines, so the structuring LLM has to reconstruct line-item tables from text alone instead of "seeing" the table
- ❌ No GPU on the dev machine this was tested on - OCR latency would likely drop substantially with one, but that's unverified here

**Vision LLM (default)**
- ✅ 5-9x faster per call in both languages tested
- ✅ Sees the actual layout/table structure directly - no OCR-order-of-lines problem
- ✅ Zero extra dependencies, works out of the box
- ❌ Missed a field on the one non-Latin-script sample tested (Tamil `tax`)
- ❌ Latency varies under concurrent batch load (Groq queuing/rate limits), not just per-call time
- ❌ Sends the full image to a third-party API on every call

**To enable PaddleOCR + Text LLM**, install it into a **separate virtual environment** - not
your system/global Python. Installing it globally during development caused real version
conflicts with unrelated packages (`langchain`, `tensorflow`, `kubernetes`) on the dev machine
this was built on:

```bash
python -m venv .venv-ocr
source .venv-ocr/bin/activate   # .venv-ocr\Scripts\activate on Windows
pip install -r requirements.txt -r requirements-ocr.txt
streamlit run enhanced_ui.py
```

If `paddleocr` isn't installed, the app detects that at import time and simply hides the
option in the UI rather than crashing - see `ocr_hybrid.py`'s `PADDLEOCR_AVAILABLE` flag.

This isn't a fully-resolved comparison — two samples (one Tamil, one English) and a handful of
timed runs is a spike, not a benchmark. See [Future Enhancements](#future-enhancements) for
turning this into one.

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
   Optionally, to enable the PaddleOCR + Text LLM extraction mode, install
   `requirements-ocr.txt` **into a separate virtual environment** - see
   [OCR Architecture](#ocr-architecture) for why that isolation matters.

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
yet — see [Future Enhancements](#future-enhancements).

##  Multilingual Support

The language picker tells the vision model what language to expect on the invoice; it does
**not** translate the extracted fields — vendor names, line-item descriptions, etc. come back
in their original script (e.g. Tamil text stays Tamil, not transliterated or translated to
English). This is extraction-in-context, not a translation feature.

Extraction accuracy on non-Latin scripts (Tamil, Arabic) depends on the underlying vision
model and tends to be lower than on Latin-script invoices, especially for numeric fields the
model might try to compute rather than read verbatim — the extraction prompt explicitly
instructs the model to report the total exactly as printed rather than recalculating it from
subtotal + tax, which fixed a real mismatch we hit on a sample Tamil invoice, but this remains
an LLM accuracy limitation, not a solved problem. Always spot-check extracted numeric fields
against the source document.

##  Future Enhancements

- [ ] **Persistence** — invoices currently live only in `st.session_state` and vanish on
      reload; add a SQLite/Postgres store so extracted history survives across sessions
- [ ] **Multi-page PDF extraction** — currently one page per PDF is extracted; merge line
      items across all pages of a multi-page invoice
- [ ] **Proper OCR-path benchmark** — the [OCR Architecture](#ocr-architecture) comparison is
      a spike (one Tamil sample, a handful of timed runs), not a benchmark; running both paths
      across all of `.Dataset/` with repeated trials would turn "here's what I observed" into
      real numbers with variance, and is the natural next step for whoever wants to trust the
      comparison table there
- [ ] **Fraud score calibration** — the signal weights and ELA threshold are hand-set
      heuristics; calibrating them against real labeled fraud examples (if any become
      available) would make the score meaningfully more trustworthy
- [ ] Additional fraud rules: round-amount detection, after-hours invoice dates,
      rapid-succession invoices from the same vendor
- [ ] Vendor reputation scoring system
- [ ] Approval workflow / status tracking (pending → approved → paid)
- [ ] Multi-currency conversion and rollup reporting
- [ ] Email ingestion (forward invoices to a monitored inbox for auto-processing)
- [ ] Predictive analytics for payment delays
- [ ] Mobile app with camera integration
- [ ] Automated benchmark harness (planted-defect scoring against ground truth) so extraction
      accuracy claims can be backed by a number instead of asserted

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
