import streamlit as st
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import plotly.express as px

# 1. Setup & Session State
st.set_page_config(page_title="Financial Agent Chat", layout="wide")
st.title("💬 Financial Agent Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history
if "context_data" not in st.session_state:
    st.session_state.context_data = "" # Store extracted text/stats

# 2. Sidebar: Auth & Model Discovery
with st.sidebar:
    st.header("🔑 Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    available_models = ["gemini-1.5-flash"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model_list = [m.name.replace('models/', '') for m in genai.list_models() 
                          if 'generateContent' in m.supported_generation_methods]
            if model_list: available_models = model_list
        except: pass

    model_id = st.selectbox("Select Model", available_models)
    
    st.divider()
    st.header("📁 Upload Knowledge")
    pdf_file = st.file_uploader("Upload PDF Report", type="pdf")
    csv_file = st.file_uploader("Upload CSV Data", type="csv")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 3. Process Uploads into Context
if pdf_file:
    reader = PdfReader(pdf_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    st.session_state.context_data = f"PDF Content: {text[:15000]}"
    st.sidebar.success("PDF Context Loaded!")

if csv_file:
    df = pd.read_csv(csv_file)
    st.session_state.context_data += f"\nCSV Stats: {df.describe().to_string()}"
    st.sidebar.success("CSV Stats Loaded!")
    with st.sidebar.expander("View Data"):
        st.dataframe(df.head())

# 4. Chat Interface
# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me about your financial data..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        if not api_key:
            response_text = "Please provide an API key in the sidebar to chat."
        else:
            try:
                model = genai.GenerativeModel(model_id)
                # Combine history + context + prompt
                full_prompt = f"System Context: {st.session_state.context_data}\n\nUser: {prompt}"
                
                # Check for visualization intent
                if any(x in prompt.lower() for x in ["plot", "chart", "visualize"]) and csv_file:
                    st.plotly_chart(px.line(df, title="Financial Visual Analysis"))
                    response_text = "I've generated the chart based on your request above."
                else:
                    response = model.generate_content(full_prompt)
                    response_text = response.text
            except Exception as e:
                response_text = f"Error: {str(e)}"
        
        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
