import streamlit as st
import pandas as pd
import whisper
import PyPDF2
import google.generativeai as genai
import os

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="FinSight Omni", layout="wide")
st.title("🚀 FinSight Omni: Multimodal Financial Agent")

# --- 2. SIDEBAR FOR API KEY & FILE UPLOAD ---
with st.sidebar:
    st.header("Settings")
    # Using st.text_input to get the API Key securely from the user
    user_api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.divider()
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload Audio, PDF, or CSV", type=["pdf", "csv", "mp3", "wav"])
    
    if st.button("Clear Conversation"):
        st.session_state.context_data = ""
        st.rerun()

# --- 3. CONFIGURING THE MODEL ---
if user_api_key:
    try:
        genai.configure(api_key=user_api_key)
        # CHANGED: Using 'gemini-flash-latest' to avoid 404 errors
        model = genai.GenerativeModel('gemini-flash-latest') 
    except Exception as e:
        st.sidebar.error("Failed to initialize API. Check your key.")

# --- 4. DATA PROCESSING LOGIC ---
if "context_data" not in st.session_state:
    st.session_state.context_data = ""

if uploaded_file and user_api_key:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner(f"Analyzing {uploaded_file.name}..."):
        if file_type in ["mp3", "wav"]:
            # Use 'tiny' model for faster local transcription
            w_model = whisper.load_model("tiny")
            with open("temp_audio", "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = w_model.transcribe("temp_audio")
            st.session_state.context_data = result['text']
            st.audio(uploaded_file)
            os.remove("temp_audio")

        elif file_type == "pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in reader.pages])
            st.session_state.context_data = text

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10))
            # Summarize to avoid token limit issues
            summary = f"Columns: {list(df.columns)}\nStats: {df.describe().to_string()}\nSample: {df.head(10).to_string()}"
            st.session_state.context_data = summary

    st.success(f"{file_type.upper()} Loaded successfully!")

# --- 5. CHAT INTERFACE ---
st.divider()
query = st.chat_input("Ask a question about your uploaded financial data...")

if query:
    if not user_api_key:
        st.error("Missing API Key in sidebar!")
    elif not st.session_state.context_data:
        st.info("Please upload a file first.")
    else:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            # Truncate context to stay within safe limits
            safe_context = st.session_state.context_data[:15000]
            
            prompt = f"CONTEXT:\n{safe_context}\n\nQUESTION: {query}"
            
            try:
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                # Catching and displaying the "Invalid Key" or "Quota Exceeded" errors clearly
                st.error(f"❌ API Error: {str(e)}")
                if "API_KEY_INVALID" in str(e):
                    st.info("💡 Double-check the API Key you entered in the sidebar.")
