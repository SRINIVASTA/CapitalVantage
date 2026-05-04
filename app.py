import streamlit as st
import pandas as pd
import whisper
import PyPDF2
import google.generativeai as genai
import os

# --- INITIAL SETUP ---
st.set_page_config(page_title="GenAI Financial Agent", layout="wide")
st.title("🎤 PDF, 📊 CSV, & 🔊 Audio Financial Agent")

# Securely configure Gemini
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # Replace with your real key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash') # Using 1.5 for larger context

# --- SIDEBAR: UPLOAD SECTION ---
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload financial file", type=["pdf", "csv", "mp3", "wav", "m4a"])
    st.info("The agent will automatically switch logic based on file type.")

# --- SHARED STATE FOR CONTEXT ---
if "context_data" not in st.session_state:
    st.session_state.context_data = ""

# --- PROCESSING LOGIC ---
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    with st.status(f"Processing {file_type.upper()}...", expanded=True) as status:
        
        # 1. HANDLE AUDIO (Whisper)
        if file_type in ["mp3", "wav", "m4a"]:
            st.write("Transcribing audio with Whisper...")
            w_model = whisper.load_model("tiny") # 'tiny' is fastest for testing
            with open("temp_audio", "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = w_model.transcribe("temp_audio")
            st.session_state.context_data = f"AUDIO TRANSCRIPT: {result['text']}"
            st.audio(uploaded_file)
            
        # 2. HANDLE PDF (RAG-style extraction)
        elif file_type == "pdf":
            st.write("Extracting PDF text...")
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            st.session_state.context_data = f"DOCUMENT CONTENT: {text}"
            
        # 3. HANDLE CSV (Data Analysis)
        elif file_type == "csv":
            st.write("Reading financial spreadsheet...")
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(5))
            # Feed Gemini the column info and summary stats
            csv_summary = f"Columns: {list(df.columns)} \n Stats: {df.describe().to_string()}"
            st.session_state.context_data = f"CSV DATA SUMMARY: {csv_summary} \n FULL DATA: {df.to_string()[:5000]}"
        
        status.update(label="File Processed Successfully!", state="complete")

# --- CHAT INTERFACE ---
st.divider()
query = st.chat_input("Ask a question about your uploaded financial data...")

if query:
    if not st.session_state.context_data:
        st.warning("Please upload a file first!")
    else:
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
            # The "Agentic" Prompt
            full_prompt = f"""
            You are a Financial AI Agent. Use the following context to answer the user:
            
            CONTEXT: {st.session_state.context_data}
            
            USER QUESTION: {query}
            
            If it's a CSV, provide numerical insights. If it's Audio or PDF, provide a summary.
            """
            response = model.generate_content(full_prompt)
            st.markdown(response.text)

# Clean up temp files
if os.path.exists("temp_audio"):
    os.remove("temp_audio")
