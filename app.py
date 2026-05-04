import streamlit as st
import pandas as pd
import whisper
import PyPDF2
import google.generativeai as genai
import os

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="FinSight Omni", layout="wide")
st.title("🚀 FinSight Omni: Multimodal Financial Agent")

# Replace with your actual API Key
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

# Use 1.5-Flash for speed and large context handling
model = genai.GenerativeModel('gemini-1.5-flash')

if "context_data" not in st.session_state:
    st.session_state.context_data = ""

# --- 2. SIDEBAR UPLOAD ---
with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload Audio, PDF, or CSV", type=["pdf", "csv", "mp3", "wav"])
    if st.button("Clear Context"):
        st.session_state.context_data = ""
        st.rerun()

# --- 3. ALL-IN-ONE PROCESSING LOGIC ---
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner(f"Analyzing {uploaded_file.name}..."):
        # AUDIO PROCESSING
        if file_type in ["mp3", "wav"]:
            w_model = whisper.load_model("tiny")
            with open("temp_audio", "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = w_model.transcribe("temp_audio")
            st.session_state.context_data = result['text']
            st.audio(uploaded_file)
            os.remove("temp_audio")

        # PDF PROCESSING
        elif file_type == "pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                content = page.extract_text()
                if content: text += content
            st.session_state.context_data = text

        # CSV PROCESSING
        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10))
            # Summarize CSV to avoid "InvalidArgument" size errors
            summary = f"Columns: {list(df.columns)}\nStats: {df.describe().to_string()}\nData Snippet: {df.head(20).to_string()}"
            st.session_state.context_data = summary

    st.success(f"{file_type.upper()} Loaded!")

# --- 4. THE AGENT INTERFACE ---
st.divider()
query = st.chat_input("Ask me about the financial data...")

if query:
    if not st.session_state.context_data:
        st.info("Please upload a file to begin.")
    else:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            # Truncate context to ~15,000 chars to avoid API "InvalidArgument" errors
            safe_context = st.session_state.context_data[:15000]
            
            prompt = f"""
            You are a Financial Expert Agent. Use the context below to answer.
            If the data is a CSV, focus on trends and numbers.
            If it is Audio or PDF, focus on summaries and key takeaways.
            
            CONTEXT:
            {safe_context}
            
            USER QUESTION:
            {query}
            """
            
            try:
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"API Error: {e}. Try a smaller file or clear context.")

