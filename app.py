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

# --- 2. SIDEBAR: DYNAMIC MODEL SELECTOR & CONFIG ---
with st.sidebar:
    st.title("⚙️ Agent Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    # Dynamic Model Loading
    available_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Fetching all active models that support content generation
            fetched_models = [m.name.replace('models/', '') for m in genai.list_models() 
                             if 'generateContent' in m.supported_generation_methods]
            if fetched_models:
                available_models = fetched_models
        except Exception:
            st.warning("Could not fetch models. Using defaults.")

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

# --- 3. CORE LOGIC: DOCUMENT INTELLIGENCE ---
pdf_text = ""
if pdf_file:
    reader = PdfReader(pdf_file)
    pdf_text = "".join([page.extract_text() or "" for page in reader.pages])
    st.sidebar.success("PDF Indexed")

if csv_file:
    st.session_state.extracted_df = pd.read_csv(csv_file)
    st.sidebar.success("CSV Loaded")

# --- 4. AGENTIC FEATURE: PROACTIVE SUGGESTIONS ---
if (pdf_text or st.session_state.extracted_df is not None) and api_key and not st.session_state.suggestions:
    with st.spinner("Agent analyzing data for insights..."):
        model = genai.GenerativeModel(model_id)
        context = pdf_text[:5000] if pdf_text else str(st.session_state.extracted_df.head())
        prompt = f"Based on this financial data, suggest 3 short analytical questions for a CFO. Return ONLY the questions.\n\nData: {context}"
        try:
            res = model.generate_content(prompt)
            st.session_state.suggestions = [q.strip() for q in res.text.split('\n') if q.strip()][:3]
        except:
            st.session_state.suggestions = ["Summarize key trends", "Analyze risks", "Show revenue chart"]

# Display clickable suggestions in sidebar
if st.session_state.suggestions:
    st.sidebar.subheader("💡 Suggested for You")
    for q in st.session_state.suggestions:
        if st.sidebar.button(q):
            st.session_state.user_input = q 

# --- 5. MAIN CHAT INTERFACE ---
st.title("💬 GenAI Financial Agent")
st.info(f"Currently active: **{model_id}** | Handling Document Intelligence & RAG")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about the data or type 'plot a chart'...") or st.session_state.get("user_input"):
    if st.session_state.get("user_input"):
        prompt = st.session_state.user_input
        del st.session_state.user_input

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("API Key required.")
        else:
            try:
                model = genai.GenerativeModel(model_id)
                
                # BRANCH A: VISUALIZATION & TABLE EXTRACTION
                if any(x in prompt.lower() for x in ["chart", "plot", "graph", "pie", "visualize"]):
                    if st.session_state.extracted_df is None and pdf_text:
                        with st.spinner("Extracting table from PDF..."):
                            extract_cmd = f"Extract the main financial table from this text as CSV. ONLY CSV code:\n{pdf_text[:10000]}"
                            raw_csv = model.generate_content(extract_cmd).text
                            clean_csv = raw_csv.replace("```csv", "").replace("```", "").strip()
                            st.session_state.extracted_df = pd.read_csv(io.StringIO(clean_csv))
                    
                    df = st.session_state.extracted_df
                    if df is not None:
                        num_cols = df.select_dtypes(include=['number']).columns.tolist()
                        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                        
                        if "pie" in prompt.lower() and cat_cols and num_cols:
                            fig = px.pie(df, names=cat_cols[0], values=num_cols[0])
                        else:
                            fig = px.bar(df, x=cat_cols[0] if cat_cols else None, y=num_cols[0] if num_cols else None)
                        
                        st.plotly_chart(fig)
                        st.dataframe(df.head())
                        res_text = "I've processed the data and generated the visualization."
                    else:
                        res_text = "I need a CSV or a PDF with tables to create a chart."

                # BRANCH B: STANDARD RAG CHAT
                else:
                    full_context = f"Context: {pdf_text[:12000]}\n\nQuestion: {prompt}"
                    response = model.generate_content(full_context)
                    res_text = response.text

                st.markdown(res_text)
                st.session_state.messages.append({"role": "assistant", "content": res_text})

            except Exception as e:
                st.error(f"Execution Error: {e}")
