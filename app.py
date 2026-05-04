import streamlit as st
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import plotly.express as px
import os

# 1. Setup & Configuration
st.set_page_config(page_title="GenAI Financial Agent", page_icon="🚀", layout="wide")

st.markdown("""
    # 🚀 GenAI Financial Agent
    Analyze PDFs and CSVs with Agentic Logic
""")

# Sidebar for API Key and Model Selection
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Enter Google Gemini API Key", type="password", help="Get yours at ://google.com")
    
    st.divider()
    st.header("🤖 Model Settings")
    # Using 1.5-flash as default to avoid 'NotFound' errors associated with 'gemini-pro'
    model_choice = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])

# 2. Initialize Gemini
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_choice)
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")

# 3. File Uploaders
col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("📂 Upload Financial Report (PDF)", type="pdf")
with col2:
    csv_file = st.file_uploader("📊 Upload Financial Data (CSV)", type="csv")

# 4. Logic & Utility Functions
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def agent_query(prompt, context=""):
    full_prompt = f"""
    You are a Financial AI Agent. Use the following context to answer the user request.
    If the context isn't enough, use your general financial knowledge but state it clearly.
    
    Context: {context}
    
    User Request: {prompt}
    """
    response = model.generate_content(full_prompt)
    return response.text

# 5. Main Application Interface
if not api_key:
    st.info("👈 Please enter your Gemini API key in the sidebar to start.")
else:
    tab1, tab2 = st.tabs(["📄 Document Summarizer", "📈 Data Analyst Agent"])

    # --- Tab 1: PDF Summarization ---
    with tab1:
        if pdf_file:
            st.success("PDF Uploaded Successfully!")
            if st.button("Generate AI Executive Summary"):
                with st.spinner("Agent is reading the document..."):
                    raw_text = get_pdf_text(pdf_file)
                    # Pass the first 15,000 characters to stay within basic prompt limits
                    summary = agent_query("Provide a concise executive summary focusing on revenue, net income, and risk factors.", context=raw_text[:15000])
                    st.subheader("Executive Summary")
                    st.markdown(summary)
        else:
            st.info("Upload a financial PDF (like an Annual Report) to see the agent in action.")

    # --- Tab 2: CSV Data Analysis ---
    with tab2:
        if csv_file:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head(10), use_container_width=True)
            
            user_query = st.text_input("Ask the Data Agent (e.g., 'What is the correlation between revenue and spend?' or 'Plot the quarterly growth')")
            
            if user_query:
                # Intent Routing Logic
                if any(word in user_query.lower() for word in ["plot", "chart", "graph", "visualize"]):
                    st.info("Agent Intent: Data Visualization")
                    numeric_df = df.select_dtypes(include=['number'])
                    if not numeric_df.empty:
                        fig = px.line(df, title="Automated Financial Trend Analysis")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No numeric columns found to visualize.")
                else:
                    with st.spinner("Agent is crunching numbers..."):
                        # Send statistical summary as context
                        data_summary = df.describe().to_string()
                        answer = agent_query(user_query, context=f"Dataset Statistics:\n{data_summary}")
                        st.markdown("### Agent Response")
                        st.write(answer)
        else:
            st.info("Upload a CSV file (e.g., sales_data.csv) to enable the Data Analyst.")

