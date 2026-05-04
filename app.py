import streamlit as st
import pandas as pd
from google import genai
from PyPDF2 import PdfReader
import plotly.express as px

# 1. Setup & Configuration
st.set_page_config(page_title="GenAI Financial Agent", page_icon="🚀", layout="wide")

st.markdown("# 🚀 GenAI Financial Agent")

# Sidebar for API Key
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.divider()
    st.header("🤖 Model Settings")
    # Using the newest stable model identifier
    model_id = st.selectbox("Select Model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])

# 2. Initialize Client
client = None
if api_key:
    try:
        # Using the new Google GenAI SDK Client
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Initialization Error: {e}")

# 3. File Uploaders
col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("📂 Upload Financial Report (PDF)", type="pdf")
with col2:
    csv_file = st.file_uploader("📊 Upload Financial Data (CSV)", type="csv")

# 4. Helper Functions
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content: text += content
    return text

def agent_query(prompt, context=""):
    full_content = f"Context: {context}\n\nUser Request: {prompt}"
    # New SDK syntax: client.models.generate_content
    response = client.models.generate_content(
        model=model_id,
        contents=full_content
    )
    return response.text

# 5. Application Logic
if not api_key:
    st.warning("Please enter your API key in the sidebar to begin.")
else:
    tab1, tab2 = st.tabs(["📄 Document Summarizer", "📈 Data Analyst Agent"])

    with tab1:
        if pdf_file:
            if st.button("Generate Summary"):
                with st.spinner("Analyzing document..."):
                    raw_text = get_pdf_text(pdf_file)
                    # Simple RAG: passing text slice as context
                    summary = agent_query("Summarize financial performance and risks.", context=raw_text[:15000])
                    st.markdown(summary)
        else:
            st.info("Upload a PDF to start.")

    with tab2:
        if csv_file:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head(), use_container_width=True)
            
            user_query = st.text_input("Ask the Data Agent:")
            if user_query:
                if any(x in user_query.lower() for x in ["plot", "chart", "visualize"]):
                    num_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if num_cols:
                        fig = px.line(df, title="Data Trend")
                        st.plotly_chart(fig)
                else:
                    with st.spinner("Thinking..."):
                        stats = df.describe().to_string()
                        answer = agent_query(user_query, context=f"Data Stats: {stats}")
                        st.write(answer)
