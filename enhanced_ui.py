import streamlit as st
import json
import base64
import pandas as pd
from utils import InvoiceData, GroqClient, preprocess_image, process_image_upload, process_image_url, display_image_preview, setup_page, show_extraction_button, display_results, display_error, run_chatbot, edit_invoice_data, export_to_csv, is_pdf_file, get_pdf_page_count, render_pdf_page
from analytics import analyze_invoices, detect_anomalies
from uuid import uuid4
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
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {accent}22;
            border-bottom: 2px solid {accent};
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

# Fraud detection
def detect_fraud(invoices):
    """Detect potential fraud in invoices using rules."""
    if not invoices:
        st.warning("No invoices to analyze for fraud.")
        return
    
    fraud_data = []
    invoice_numbers = [inv["invoice"].invoice_number for inv in invoices if inv["invoice"].invoice_number]
    duplicates = {num for num in invoice_numbers if invoice_numbers.count(num) > 1}
    
    for inv in invoices:
        invoice = inv["invoice"]
        flags = []
        if invoice.invoice_number in duplicates:
            flags.append("Duplicate invoice number detected.")
        if invoice.total_amount and invoice.total_amount > 100000:
            flags.append("Unusually high total amount.")
        if invoice.tax and invoice.total_amount and invoice.tax > 0.3 * invoice.total_amount:
            flags.append("Unusually high tax amount.")
        if flags:
            fraud_data.append({
                "Invoice ID": inv["image_id"],
                "Invoice Number": invoice.invoice_number,
                "Total Amount": invoice.total_amount,
                "Tax": invoice.tax,
                "Flags": "; ".join(flags)
            })
    
    if fraud_data:
        st.subheader("Potential Fraud Alerts")
        st.dataframe(pd.DataFrame(fraud_data))
    else:
        st.success("No potential fraud detected.")

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
                    for i, (image_bytes, mime_type) in enumerate(zip(image_bytes_list, mime_types)):
                        with st.spinner(f"Extracting data from image {i+1}..."):
                            try:
                                groq_client = GroqClient(api_key=st.session_state.groq_api_key)
                                # Dynamic OCR prompt
                                initial_prompt = """
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
                                Return the result strictly in JSON format with 'data' and 'confidence_scores' keys.
                                """
                                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                                image_content = {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                                }
                                initial_data = groq_client.extract_invoice_data(
                                    initial_prompt.format(
                                        language=language,
                                        schema=json.dumps(InvoiceData.model_json_schema(), indent=2)
                                    ),
                                    image_content
                                )
                                invoice_type = detect_invoice_type(initial_data.get("data", {}))
                                
                                type_specific_prompts = {
                                    "retail": "Focus on product SKUs, quantities, and unit prices in line items.",
                                    "service": "Emphasize service descriptions, hours worked, and rates in line items.",
                                    "utility": "Prioritize billing periods, meter readings, and rate structures.",
                                    "general": "Extract all fields as per the schema."
                                }
                                prompt = initial_prompt + f"\nSpecific instructions for {invoice_type} invoices: {type_specific_prompts[invoice_type]}"
                                
                                max_retries = 2
                                for attempt in range(max_retries):
                                    try:
                                        extracted_data = groq_client.extract_invoice_data(
                                            prompt.format(
                                                language=language,
                                                schema=json.dumps(InvoiceData.model_json_schema(), indent=2)
                                            ),
                                            image_content
                                        )
                                        invoice = InvoiceData(**extracted_data.get("data", {}))
                                        confidence_scores = extracted_data.get("confidence_scores", {})
                                        
                                        if all(value is None for value in extracted_data.get("data", {}).values()):
                                            st.warning(f"Image {i+1}, Attempt {attempt + 1}: No data extracted. Retrying..." if attempt < max_retries - 1 else f"Image {i+1}: All attempts failed.")
                                            continue
                                        
                                        st.session_state.invoices.append({
                                            "invoice": invoice,
                                            "confidence_scores": confidence_scores,
                                            "image_id": str(uuid4()),
                                            "invoice_type": invoice_type
                                        })
                                        display_results(invoice)
                                        st.subheader("Confidence Scores")
                                        st.json(confidence_scores)
                                        st.info(f"Detected Invoice Type: {invoice_type.capitalize()}")
                                        st.success(f"Image {i+1} processed successfully!")
                                        break
                                    
                                    except Exception as e:
                                        if attempt < max_retries - 1:
                                            st.warning(f"Image {i+1}, Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                                            continue
                                        display_error(f"Image {i+1}: Failed to parse after {max_retries} attempts: {str(e)}")
                            
                            except Exception as e:
                                display_error(f"Image {i+1}: Failed to parse: {str(e)}. Try a clearer image.")
                        
                        progress_bar.progress((i + 1) / len(image_bytes_list))
                
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
        with st.container():
            predefined_prompts = [
                "Summarize the latest invoice",
                "Check for missing fields in invoices",
                "List all vendors",
                "What is the total amount of all invoices?"
            ]
            selected_prompt = st.selectbox("Quick Questions", [""] + predefined_prompts, key="predefined_prompt")
            
            invoice_context = json.dumps([inv["invoice"].dict() for inv in st.session_state.invoices], indent=2)
            user_input = st.text_input("Ask a question about your invoices:", key="chat_input")
            
            if user_input or selected_prompt:
                prompt = selected_prompt or user_input
                full_prompt = f"""
                You are an invoice processing assistant. Use the following invoice data as context:
                {invoice_context}
                Answer the user's question: {prompt}
                Provide a concise, accurate response. If the question is unrelated to invoices, politely redirect to invoice-related queries.
                """
                try:
                    groq_client = GroqClient(api_key=st.session_state.groq_api_key)
                    response = groq_client.run_chatbot_query(full_prompt)
                    st.session_state.chat_history.append({"user": prompt, "bot": response})
                    st.markdown("**Response:**")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Chatbot error: {str(e)}. Please try again.")
            
            st.sidebar.subheader("Chat History")
            for i, chat in enumerate(st.session_state.chat_history):
                with st.sidebar.expander(f"Chat {i+1}"):
                    st.write(f"**You:** {chat['user']}")
                    st.write(f"**Bot:** {chat['bot']}")

    with tab3:
        st.header("Fraud Detection")
        if st.session_state.invoices:
            detect_fraud(st.session_state.invoices)
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