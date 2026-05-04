import streamlit as st
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import plotly.express as px

# 1. Setup & Configuration
st.set_page_config(page_title="GenAI Financial Agent", layout="wide")
st.title("🚀 GenAI Financial Agent")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')

# 2. File Uploaders
col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("Upload Financial Report (PDF)", type="pdf")
with col2:
    csv_file = st.file_uploader("Upload Financial Data (CSV)", type="csv")

# 3. Helper Functions
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def agent_query(prompt, context=""):
    full_prompt = f"Context: {context}\n\nUser Request: {prompt}"
    response = model.generate_content(full_prompt)
    return response.text

# 4. Main Agent Logic
if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to begin.")
else:
    mode = st.tabs(["Document Summarizer", "Data Analyst Agent"])

    # Tab 1: Summarization Agent
    with mode[0]:
        if pdf_file:
            if st.button("Generate Executive Summary"):
                with st.spinner("Extracting and analyzing..."):
                    raw_text = get_pdf_text(pdf_file)
                    # Simple RAG: passing first 10k chars as context
                    summary = agent_query("Summarize the key financial metrics and risks.", context=raw_text[:10000])
                    st.subheader("Executive Summary")
                    st.write(summary)
        else:
            st.info("Upload a PDF to use the Summarizer.")

    # Tab 2: Data & Visualization Agent
    with mode[1]:
        if csv_file:
            df = pd.read_csv(csv_file)
            st.write("Data Preview:", df.head())
            
            user_query = st.text_input("Ask about the data (e.g., 'Visualize the revenue trend' or 'What is the avg profit?')")
            
            if user_query:
                # Basic Intent Routing
                if any(word in user_query.lower() for word in ["chart", "plot", "graph", "visualize"]):
                    st.info("Agent: Generating Visualization...")
                    num_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if num_cols:
                        fig = px.line(df, y=num_cols[0], title=f"Trend Analysis: {num_cols[0]}")
                        st.plotly_chart(fig)
                else:
                    with st.spinner("Agent is thinking..."):
                        stats = df.describe().to_string()
                        answer = agent_query(user_query, context=f"Dataset Stats: {stats}")
                        st.write(answer)
        else:
            st.info("Upload a CSV to use the Data Analyst Agent.")
