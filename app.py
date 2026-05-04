import streamlit as st
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import plotly.express as px
import io

# 1. Setup & Session State
st.set_page_config(page_title="Financial Agent Chat", page_icon="💰", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_df" not in st.session_state:
    st.session_state.extracted_df = None # Stores data extracted from PDF or uploaded CSV

# 2. Sidebar: Configuration
with st.sidebar:
    st.header("🔑 API Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    available_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = [m.name.replace('models/', '') for m in genai.list_models() 
                     if 'generateContent' in m.supported_generation_methods]
            if models: available_models = models
        except:
            st.warning("Using default models.")

    model_id = st.selectbox("Select Model", available_models)
    
    st.divider()
    st.header("📁 Knowledge Base")
    pdf_file = st.file_uploader("Upload Financial PDF", type="pdf")
    csv_file = st.file_uploader("Upload Financial CSV", type="csv")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.extracted_df = None
        st.rerun()

# 3. Data Processing Logic
pdf_text = ""
if pdf_file:
    reader = PdfReader(pdf_file)
    pdf_text = "".join([page.extract_text() or "" for page in reader.pages])
    st.sidebar.success("PDF Content Indexed")

if csv_file:
    st.session_state.extracted_df = pd.read_csv(csv_file)
    st.sidebar.success("CSV Data Loaded")

# 4. Chat UI
st.title("💬 GenAI Financial Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or request a chart..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please provide a Gemini API key.")
        else:
            try:
                model = genai.GenerativeModel(model_id)
                
                # --- AUTO-CONVERT PDF TO DATA FRAME IF CHART REQUESTED ---
                if any(x in prompt.lower() for x in ["chart", "plot", "pie", "graph"]):
                    if st.session_state.extracted_df is None and pdf_text:
                        with st.spinner("Agent extracting table data from PDF..."):
                            # Logic: Ask Gemini to extract a CSV string from the PDF text
                            extract_prompt = f"""
                            Extract the main financial table from this text as a raw CSV string. 
                            Only return the CSV data, no conversation. Use commas as delimiters.
                            
                            TEXT: {pdf_text[:15000]}
                            """
                            raw_csv = model.generate_content(extract_prompt).text
                            # Clean response to ensure it's valid CSV
                            clean_csv = raw_csv.replace("```csv", "").replace("```", "").strip()
                            st.session_state.extracted_df = pd.read_csv(io.StringIO(clean_csv))
                            st.success("Converted PDF data to internal table!")

                    # --- VISUALIZATION LOGIC ---
                    df = st.session_state.extracted_df
                    if df is not None:
                        num_cols = df.select_dtypes(include=['number']).columns.tolist()
                        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                        
                        if "pie" in prompt.lower() and cat_cols and num_cols:
                            fig = px.pie(df, names=cat_cols[0], values=num_cols[0], title="Financial Distribution")
                            st.plotly_chart(fig)
                        elif num_cols:
                            fig = px.bar(df, y=num_cols[0], title="Data Analysis")
                            st.plotly_chart(fig)
                        
                        st.dataframe(df.head()) # Show the table for transparency
                        response_text = "I've extracted the data and generated your chart."
                    else:
                        response_text = "I couldn't find structured data in the PDF to convert to a chart."
                
                else:
                    # Standard RAG Chat
                    full_prompt = f"Context: {pdf_text[:10000]}\nUser: {prompt}"
                    response = model.generate_content(full_prompt)
                    response_text = response.text

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"Error: {str(e)}")
