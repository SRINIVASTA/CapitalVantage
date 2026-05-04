import streamlit as st
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import plotly.express as px
import io

# --- 1. CONFIGURATION & SESSION STATE ---
st.set_page_config(page_title="EY GenAI Financial Agent", page_icon="💰", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_df" not in st.session_state:
    st.session_state.extracted_df = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# --- 2. SIDEBAR: DYNAMIC CONFIG & MODELS ---
with st.sidebar:
    st.title("⚙️ Agent Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    # Model Selector Logic
    available_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            fetched_models = [m.name.replace('models/', '') for m in genai.list_models() 
                             if 'generateContent' in m.supported_generation_methods]
            if fetched_models:
                available_models = fetched_models
        except Exception:
            st.warning("Defaulting to standard models.")

    model_id = st.selectbox("Select Active Model", available_models)
    
    st.divider()
    st.header("📁 Knowledge Base")
    pdf_file = st.file_uploader("Upload Financial PDF", type="pdf")
    csv_file = st.file_uploader("Upload Financial CSV", type="csv")

    if st.button("Clear Chat & Memory"):
        st.session_state.messages = []
        st.session_state.extracted_df = None
        st.session_state.suggestions = []
        st.rerun()

# --- 3. DATA PROCESSING ---
pdf_text = ""
if pdf_file:
    reader = PdfReader(pdf_file)
    pdf_text = "".join([page.extract_text() or "" for page in reader.pages])
    st.sidebar.success("PDF Indexed")

if csv_file:
    st.session_state.extracted_df = pd.read_csv(csv_file)
    st.sidebar.success("CSV Loaded")

# --- 4. AGENTIC SUGGESTIONS (PROACTIVE ANALYSIS) ---
if (pdf_text or st.session_state.extracted_df is not None) and api_key and not st.session_state.suggestions:
    with st.spinner("Analyzing context..."):
        model = genai.GenerativeModel(model_id)
        context_preview = pdf_text[:4000] if pdf_text else str(st.session_state.extracted_df.head())
        s_prompt = f"Suggest 3 high-level financial questions based on this data. Return only questions:\n{context_preview}"
        try:
            res = model.generate_content(s_prompt)
            st.session_state.suggestions = [q.strip() for q in res.text.split('\n') if q.strip()][:3]
        except:
            st.session_state.suggestions = ["Summarize Trends", "Risk Analysis", "Project ROI"]

if st.session_state.suggestions:
    st.sidebar.subheader("💡 Suggestions")
    for q in st.session_state.suggestions:
        if st.sidebar.button(q):
            st.session_state.active_prompt = q

# --- 5. MAIN INTERFACE & CHAT LOGIC ---
st.title("💬 GenAI Financial Agent")
st.info(f"Active Model: **{model_id}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Logic (Handles keyboard input OR suggestion buttons)
user_query = st.chat_input("Ask a question...")
if st.session_state.get("active_prompt"):
    user_query = st.session_state.active_prompt
    del st.session_state.active_prompt

if user_query:
    # --- VALIDATION LAYER ---
    if not api_key:
        st.error("🔑 Please enter an API key to proceed.")
    elif not pdf_text and st.session_state.extracted_df is None:
        st.warning("⚠️ Please upload a PDF or CSV file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                model = genai.GenerativeModel(model_id)
                
                # Check for Visualization Intent
                if any(k in user_query.lower() for k in ["chart", "plot", "graph", "visualize"]):
                    if st.session_state.extracted_df is None and pdf_text:
                        with st.spinner("Converting PDF table to Data..."):
                            e_prompt = f"Extract the table from this text as CSV only:\n{pdf_text[:8000]}"
                            csv_data = model.generate_content(e_prompt).text.replace("```csv", "").replace("```", "").strip()
                            st.session_state.extracted_df = pd.read_csv(io.StringIO(csv_data))
                    
                    if st.session_state.extracted_df is not None:
                        df = st.session_state.extracted_df
                        num_cols = df.select_dtypes(include=['number']).columns.tolist()
                        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                        fig = px.bar(df, x=cat_cols[0] if cat_cols else None, y=num_cols[0] if num_cols else None)
                        st.plotly_chart(fig)
                        ans = "Data extracted and visualized successfully."
                    else:
                        ans = "I couldn't find a table to visualize."
                
                # Default RAG Chat
                else:
                    full_prompt = f"Context:\n{pdf_text[:12000]}\n\nUser Question: {user_query}"
                    ans = model.generate_content(full_prompt).text

                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})

            except Exception as e:
                st.error(f"Error processing request: {e}")
