import streamlit as st
import json
import base64
import pandas as pd
from utils import InvoiceData, GroqClient, preprocess_image, process_image_upload, process_image_url, display_image_preview, setup_page, show_extraction_button, display_results, display_error, run_chatbot, edit_invoice_data, export_to_csv, is_pdf_file, get_pdf_page_count, render_pdf_page
from analytics import analyze_invoices, detect_anomalies, score_invoices
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# ---------------------------
# Theming
# ---------------------------

def build_theme_css(theme: str) -> str:
    """Return a <style> block using concrete colors for the chosen theme.

    Branching in Python (rather than relying on a CSS [data-theme] selector,
    which tracks the browser/OS theme and not this in-app picker) is what
    makes the sidebar Theme selector actually change the page.
    """
    if theme == "Dark":
        bg, surface, text, muted, border, accent = "#10131a", "#1b202b", "#eef1f6", "#9aa4b2", "#2a3140", "#6c8ef5"
    else:
        bg, surface, text, muted, border, accent = "#f7f8fb", "#ffffff", "#1a1f2b", "#5b6472", "#e3e6eb", "#4b7bec"

    return f"""
        <style>
        .stApp {{
            background-color: {bg};
        }}
        .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, h1, h2, h3, h4 {{
            color: {text} !important;
        }}
        /* Streamlit's own toolbar and file-uploader dropzone keep a fixed dark
           background regardless of app theme - the blanket span/p/label rule
           above overrides their text to the app's (light-mode) dark color,
           making them unreadable dark-on-dark. Force them back to light text.
           (`.stApp` prefix is required so this out-specifies the `.stApp span`
           rule above - a bare `[data-testid=...] *` selector loses that fight.) */
        .stApp [data-testid="stHeader"], .stApp [data-testid="stHeader"] span,
        .stApp [data-testid="stFileUploaderDropzone"],
        .stApp [data-testid="stFileUploaderDropzone"] span,
        .stApp [data-testid="stFileUploaderDropzone"] div,
        .stApp [data-testid="stFileUploaderDropzone"] small {{
            color: #fafafa !important;
        }}
        /* The selectbox's selected-value text sits in a nested div whose
           background BaseWeb's own generated utility classes control - those
           compound multi-class selectors out-specificity a plain override,
           so the :not(#_):not(#_) chain is a standard CSS trick to force
           higher specificity without hardcoding fragile auto-generated class
           names, so the text color reliably follows the current theme here too. */
        [data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div:not(#_):not(#_) {{
            color: {text} !important;
        }}
        .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
        }}
        h1 {{
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }}
        h2, h3 {{
            font-weight: 700 !important;
            margin-top: 1.5rem !important;
        }}
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {muted} !important;
        }}
        .st-expander, .stAlert, .stTextInput > div, .stSelectbox > div,
        .stFileUploader, .stDataFrame, [data-testid="stExpander"] {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 10px;
        }}
        [data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea {{
            background-color: {surface} !important;
            color: {text} !important;
            border-color: {border} !important;
        }}
        [data-baseweb="popover"] li, [data-baseweb="menu"] li {{
            background-color: {surface} !important;
            color: {text} !important;
        }}
        .stButton>button, .stDownloadButton>button {{
            background-color: {accent};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.25rem;
            font-weight: 600;
            transition: transform 0.15s ease, filter 0.15s ease;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            filter: brightness(1.08);
            transform: translateY(-1px);
        }}
        [data-testid="stSidebar"] {{
            background-color: {surface};
            border-right: 1px solid {border};
        }}
        [data-testid="stSidebar"] * {{
            color: {text} !important;
        }}
        [data-testid="stSidebar"] .block-container {{
            padding-top: 2rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {accent}22;
            border-bottom: 2px solid {accent};
        }}
        [data-testid="stChatMessage"] {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 12px;
        }}
        .low-confidence {{
            background-color: rgba(255, 99, 132, 0.2);
        }}
        </style>
    """

# Invoice type detection
def detect_invoice_type(invoice_data: dict) -> str:
    """Detect invoice type based on keywords in vendor name or line items."""
    keywords = {
        "retail": ["store", "shop", "mart", "sku", "product"],
        "service": ["consulting", "service", "hours", "labor", "professional"],
        "utility": ["electricity", "water", "gas", "bill", "utility"]
    }
    # `.get(key, default)` only falls back when the key is missing, not when
    # it's present with a None value (which the LLM returns for empty fields),
    # so `or` is required here to avoid a NoneType crash on those invoices.
    vendor = (invoice_data.get("vendor_name") or "").lower()
    line_items = invoice_data.get("line_items") or []
    descriptions = [(item.get("description") or "").lower() for item in line_items]

    for inv_type, kws in keywords.items():
        if any(kw in vendor for kw in kws) or any(any(kw in desc for kw in kws) for desc in descriptions):
            return inv_type
    return "general"

EXTRACTION_PROMPT_TEMPLATE = """
You are an intelligent OCR extraction agent capable of understanding and processing invoices in {language}.
Extract all relevant information from the provided invoice image in structured JSON format.
The JSON object must follow this schema: {schema}.
Include a confidence score (0.0 to 1.0) for each extracted field in a separate 'confidence_scores' object.
If a field cannot be found, return it as null.
Look for common invoice patterns such as:
- Invoice number: Often labeled as 'Invoice #', 'No.', or similar.
- Dates: Look for 'Date', 'Issued', 'Due', in formats like MM/DD/YYYY or DD/MM/YYYY.
- Addresses: Look for 'Bill to', 'Ship to', or multi-line address blocks.
- Line items: Tables or lists with description, quantity, unit price, and total.
- Totals: Look for 'Subtotal', 'Tax', 'Total', often at the bottom.
- Currency: Look for symbols ($, €, £) or codes (USD, EUR).
IMPORTANT: total_amount must be the literal total figure printed on
the invoice, not a value you compute by adding subtotal and tax
yourself - if the printed total looks inconsistent with subtotal +
tax, still report the printed value verbatim; that inconsistency is
a real finding, not a fix for you to make.
Return the result strictly in JSON format with 'data' and 'confidence_scores' keys.
"""

def extract_one_invoice(image_bytes: bytes, mime_type: str, language: str, api_key: str, max_retries: int = 2):
    """Run one image through extraction and return (record, error). Runs
    inside a worker thread as part of the batch's concurrent processing, so
    it must never call any st.* function - Streamlit's session context isn't
    available off the main script thread. UI rendering happens afterward,
    back on the main thread, using the record this returns.

    This used to be two full vision-LLM calls per image: one purely to
    classify the invoice type, then a second, nearly identical call that was
    the one actually used. The type classification only needs the fields the
    single call below already extracts (vendor name, line-item descriptions),
    so the second call was pure overhead - cutting it roughly halves
    per-image latency.
    """
    groq_client = GroqClient(api_key=api_key)
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        language=language,
        schema=json.dumps(InvoiceData.model_json_schema(), indent=2)
    )
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
    }

    last_error = "No data extracted"
    for _ in range(max_retries):
        try:
            extracted_data = groq_client.extract_invoice_data(prompt, image_content)
            data = extracted_data.get("data", {})
            if all(value is None for value in data.values()):
                last_error = "Model returned no data"
                continue
            invoice = InvoiceData(**data)
            record = {
                "invoice": invoice,
                "confidence_scores": extracted_data.get("confidence_scores", {}),
                "image_id": str(uuid4()),
                "invoice_type": detect_invoice_type(data),
                "image_bytes": image_bytes,
            }
            return record, None
        except Exception as e:
            last_error = str(e)
    return None, last_error

# Fraud detection
RISK_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

def render_fraud_detection(invoices):
    """Render one expandable card per invoice: extracted fields, fraud
    score, risk category, and the specific reasons behind the score (each
    reason maps to a check score_invoices() actually ran - see analytics.py)."""
    if not invoices:
        st.warning("No invoices to analyze for fraud.")
        return

    scored = score_invoices(invoices)
    scored_by_id = {s["image_id"]: s for s in scored}

    high_count = sum(1 for s in scored if s["risk_category"] == "High")
    medium_count = sum(1 for s in scored if s["risk_category"] == "Medium")
    if high_count or medium_count:
        st.warning(f"⚠️ {high_count} high-risk and {medium_count} medium-risk invoice(s) out of {len(invoices)}.")
    else:
        st.success(f"✅ No elevated-risk invoices out of {len(invoices)} scored.")

    for inv in sorted(invoices, key=lambda i: -scored_by_id[i["image_id"]]["score"]):
        result = scored_by_id[inv["image_id"]]
        invoice = inv["invoice"]
        badge = RISK_COLORS[result["risk_category"]]
        label = f"{badge} {invoice.invoice_number or 'Unknown invoice'} — {result['risk_category']} risk (score {result['score']}/100)"
        with st.expander(label, expanded=(result["risk_category"] == "High")):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Extracted fields**")
                st.json({k: v for k, v in invoice.dict().items() if k != "line_items"})
            with col2:
                st.markdown("**Risk score**")
                st.progress(result["score"] / 100)
                st.markdown(f"**Category:** {badge} {result['risk_category']}")
                st.markdown("**Why it was flagged**")
                if result["reasons"]:
                    for reason in result["reasons"]:
                        st.markdown(f"- {reason}")
                else:
                    st.markdown("- No red flags detected.")

# Batch processing status
def display_batch_status(invoices):
    """Display summary of processed invoices."""
    total = len(invoices)
    successful = sum(1 for inv in invoices if inv["invoice"].invoice_number is not None)
    st.sidebar.subheader("Batch Processing Status")
    st.sidebar.write(f"Total Invoices: {total}")
    st.sidebar.write(f"Successfully Processed: {successful}")
    st.sidebar.write(f"Success Rate: {successful / total * 100:.1f}%" if total > 0 else "Success Rate: 0%")

def select_input_method():
    """Custom input method selection."""
    return st.radio(
        "Select input method: 📸",
        ["Upload Image 📤", "Image URL 🌐"],
        key="enhanced_input_method"
    )

def enhanced_ui():
    # Setup page
    setup_page()

    # Sidebar settings (read first so the CSS below can react to the choice)
    st.sidebar.title("Invoice AI Dashboard")
    st.sidebar.markdown("### Settings")
    language = st.sidebar.selectbox(
        "Select Invoice Language",
        ["English", "Spanish", "French", "German", "Tamil", "Other"],
        key="language_select"
    )
    dark_mode = st.sidebar.toggle("🌙 Dark mode", key="dark_mode_toggle")
    theme = "Dark" if dark_mode else "Light"

    st.markdown(build_theme_css(theme), unsafe_allow_html=True)

    # Initialize session state
    if "invoices" not in st.session_state:
        st.session_state.invoices = []
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # API key setup
    parser = argparse.ArgumentParser(description='Run the Streamlit app.')
    parser.add_argument('--environment', type=str, choices=['local', 'cloud'], default='cloud')
    args = parser.parse_args()
    
    if args.environment == 'cloud':
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except KeyError:
            st.error("GROQ_API_KEY not found in Streamlit secrets.")
            return
    else:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment variables.")
            return
    
    st.session_state.groq_api_key = groq_api_key

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Invoice Extraction", "🤖 Chatbot", "🚨 Fraud Detection", "📊 Analytics"])

    with tab1:
        st.header("Invoice Extraction")
        input_method = select_input_method()
        image_bytes_list = []
        mime_types = []
        
        with st.container():
            st.subheader("Upload Invoices")
            with st.expander("Input Options", expanded=True):
                if input_method == "Upload Image 📤":
                    uploaded_files = st.file_uploader(
                        "Upload invoice images or PDFs (supports multiple)",
                        type=["png", "jpg", "jpeg", "pdf"],
                        accept_multiple_files=True,
                        key="batch_uploader"
                    )
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            try:
                                if is_pdf_file(uploaded_file):
                                    pdf_bytes = uploaded_file.read()
                                    page_count = get_pdf_page_count(pdf_bytes)
                                    page_number = 0
                                    if page_count > 1:
                                        page_number = st.number_input(
                                            f"{uploaded_file.name} has {page_count} pages. Select page to extract:",
                                            min_value=1, max_value=page_count, value=1,
                                            key=f"pdf_page_{uploaded_file.name}"
                                        ) - 1
                                    image_bytes = render_pdf_page(pdf_bytes, page_number=page_number)
                                    mime_type = "image/png"
                                else:
                                    image_bytes, mime_type = process_image_upload(uploaded_file)
                                if image_bytes:
                                    image_bytes = preprocess_image(image_bytes)
                                    image_bytes_list.append(image_bytes)
                                    mime_types.append(mime_type)
                                    st.success(f"{uploaded_file.name} uploaded successfully!")
                            except Exception as e:
                                display_error(f"Failed to process {uploaded_file.name}: {str(e)}")
                else:
                    image_url = st.text_input(
                        "Enter image URL:",
                        key="url_input",
                        placeholder="https://example.com/invoice.jpg"
                    )
                    if image_url:
                        try:
                            image_bytes = process_image_url(image_url)
                            if image_bytes:
                                image_bytes = preprocess_image(image_bytes)
                                image_bytes_list.append(image_bytes)
                                mime_types.append("image/jpeg")
                                st.success("Image URL processed successfully!")
                        except ValueError as e:
                            display_error(str(e))
        
        if image_bytes_list:
            col1, col2 = st.columns([1, 2], gap="medium")
            with col1:
                st.subheader("Invoice Images")
                for i, image_bytes in enumerate(image_bytes_list):
                    st.write(f"Image {i+1}")
                    display_image_preview(image_bytes)
            
            with col2:
                st.subheader("Extracted Invoice Data")
                if show_extraction_button():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    n = len(image_bytes_list)
                    max_workers = min(4, n)  # Groq's per-account concurrency has a ceiling too
                    completed = 0
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(
                                extract_one_invoice, image_bytes, mime_type, language, st.session_state.groq_api_key
                            ): i
                            for i, (image_bytes, mime_type) in enumerate(zip(image_bytes_list, mime_types))
                        }
                        results = [None] * n
                        for future in as_completed(futures):
                            i = futures[future]
                            results[i] = future.result()
                            completed += 1
                            progress_bar.progress(completed / n)
                            status_text.text(f"Processed {completed}/{n} images...")
                    status_text.empty()

                    for i, (record, error) in enumerate(results):
                        if record:
                            st.session_state.invoices.append(record)
                            display_results(record["invoice"])
                            st.subheader("Confidence Scores")
                            st.json(record["confidence_scores"])
                            st.info(f"Detected Invoice Type: {record['invoice_type'].capitalize()}")
                            st.success(f"Image {i+1} processed successfully!")
                        else:
                            display_error(f"Image {i+1}: Failed to parse after retries: {error}")
                
                # Data editing with validation feedback
                if st.session_state.invoices:
                    st.subheader("Edit Invoices")
                    invoice_data = [
                        {
                            "Invoice ID": inv["image_id"],
                            "Invoice Number": inv["invoice"].invoice_number,
                            "Total Amount": inv["invoice"].total_amount,
                            "Tax": inv["invoice"].tax,
                            "Date": inv["invoice"].invoice_date,
                            "Invoice Type": inv["invoice_type"],
                            "Confidence (Invoice Number)": inv["confidence_scores"].get("invoice_number") or 1.0,
                            "Confidence (Total Amount)": inv["confidence_scores"].get("total_amount") or 1.0,
                            "Confidence (Tax)": inv["confidence_scores"].get("tax") or 1.0
                        } for inv in st.session_state.invoices
                    ]
                    def highlight_low_confidence(row):
                        styles = [""] * len(row)
                        for i, col in enumerate(row.index):
                            if "Confidence" in col and row[col] is not None and row[col] < 0.7:
                                styles[i] = "background-color: rgba(255, 99, 132, 0.2)"
                        return styles
                    
                    edited_df = st.data_editor(
                        pd.DataFrame(invoice_data),
                        column_config={
                            "Invoice ID": {"editable": False},
                            "Invoice Number": {"type": "text"},
                            "Total Amount": {"type": "number"},
                            "Tax": {"type": "number"},
                            "Date": {"type": "text"},
                            "Invoice Type": {"editable": False},
                            "Confidence (Invoice Number)": {"editable": False},
                            "Confidence (Total Amount)": {"editable": False},
                            "Confidence (Tax)": {"editable": False}
                        },
                        key="invoice_editor"
                    )
                    st.dataframe(edited_df.style.apply(highlight_low_confidence, axis=1))
                    if st.button("Save Edited Data", key="save_edit"):
                        for i, row in edited_df.iterrows():
                            for inv in st.session_state.invoices:
                                if inv["image_id"] == row["Invoice ID"]:
                                    inv["invoice"].invoice_number = row["Invoice Number"]
                                    inv["invoice"].total_amount = row["Total Amount"]
                                    inv["invoice"].tax = row["Tax"]
                                    inv["invoice"].invoice_date = row["Date"]
                        st.success("✅ Data updated successfully!")
                
                # Export options
                # Data is cheap to serialize, so the download buttons are rendered
                # directly rather than gated behind a separate trigger button -
                # wrapping st.download_button inside `if st.button(...)` requires
                # two clicks and the button vanishes on the next unrelated rerun.
                st.subheader("Export Data")
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    csv_data = export_to_csv([inv["invoice"] for inv in st.session_state.invoices])
                    st.download_button(
                        label="Download All as CSV",
                        data=csv_data,
                        file_name="all_invoices.csv",
                        mime="text/csv",
                        key="csv_download"
                    )
                with col_export2:
                    json_data = json.dumps([inv["invoice"].dict() for inv in st.session_state.invoices], indent=2)
                    st.download_button(
                        label="Download All as JSON",
                        data=json_data,
                        file_name="all_invoices.json",
                        mime="application/json",
                        key="json_download"
                    )

    with tab2:
        st.header("Invoice Assistant Chatbot")

        if not st.session_state.invoices:
            st.info("Process at least one invoice in the Extraction tab to give the assistant something to answer questions about.")

        # Replay the conversation as a real chat thread.
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat["user"])
            with st.chat_message("assistant"):
                st.markdown(chat["bot"])

        predefined_prompts = [
            "Summarize the latest invoice",
            "Check for missing fields in invoices",
            "List all vendors",
            "What is the total amount of all invoices?"
        ]
        st.caption("Quick questions")
        quick_cols = st.columns(len(predefined_prompts))
        queued_prompt = None
        for col, quick_prompt in zip(quick_cols, predefined_prompts):
            if col.button(quick_prompt, key=f"quick_{quick_prompt}", use_container_width=True):
                queued_prompt = quick_prompt

        # st.chat_input clears itself after submission and st.button only
        # returns True on the run it was clicked, so neither re-fires the
        # same question on unrelated reruns - unlike the old st.text_input +
        # st.selectbox pattern, which kept re-answering the last question on
        # every rerun anywhere else in the app.
        typed_prompt = st.chat_input("Ask a question about your invoices...")
        prompt = queued_prompt or typed_prompt

        if prompt:
            invoice_context = json.dumps([inv["invoice"].dict() for inv in st.session_state.invoices], indent=2)
            full_prompt = f"""
            You are an invoice processing assistant. Use the following invoice data as context:
            {invoice_context}
            Answer the user's question: {prompt}
            Provide a concise, accurate response. If the question is unrelated to invoices, politely redirect to invoice-related queries.
            """
            with st.chat_message("user"):
                st.markdown(prompt)
            try:
                groq_client = GroqClient(api_key=st.session_state.groq_api_key)
                with st.spinner("Thinking..."):
                    response = groq_client.run_chatbot_query(full_prompt)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.chat_history.append({"user": prompt, "bot": response})
            except Exception as e:
                st.error(f"Chatbot error: {str(e)}. Please try again.")

    with tab3:
        st.header("Fraud Detection")
        st.caption("Each invoice below is scored from the checks it actually triggered - duplicate numbers, disproportionate amounts, arithmetic inconsistency, low OCR confidence, statistical outliers, and an image-manipulation heuristic.")
        if st.session_state.invoices:
            render_fraud_detection(st.session_state.invoices)
        else:
            st.info("No invoices processed yet. Upload invoices in the Extraction tab.")

    with tab4:
        st.header("Analytics")
        if st.session_state.invoices:
            analyze_invoices(st.session_state.invoices)
            detect_anomalies(st.session_state.invoices)
        else:
            st.info("No invoices processed yet. Upload invoices in the Extraction tab.")

    # Batch processing status
    display_batch_status(st.session_state.invoices)

if __name__ == "__main__":
    enhanced_ui()