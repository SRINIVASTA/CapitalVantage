### `app.py`
import streamlit as st
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import plotly.express as px

# 1. Setup & Session State
st.set_page_config(page_title="Financial Agent Chat", page_icon="💰", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = "" # Processed PDF text

# 2. Sidebar: Configuration
with st.sidebar:
    st.header("🔑 API Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    available_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Dynamic Model Discovery
            models = [m.name.replace('models/', '') for m in genai.list_models() 
                     if 'generateContent' in m.supported_generation_methods]
            if models: available_models = models
        except:
            st.warning("Could not fetch models. Using defaults.")

    model_id = st.selectbox("Select Model", available_models)
    
    st.divider()
    st.header("📁 Knowledge Base")
    pdf_file = st.file_uploader("Upload Financial PDF", type="pdf")
    csv_file = st.file_uploader("Upload Financial CSV", type="csv")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 3. Knowledge Extraction Logic
if pdf_file:
    with st.spinner("Extracting PDF text..."):
        reader = PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        st.session_state.pdf_context = text[:15000] # Limit to 15k chars for prompt safety
        st.sidebar.success("PDF Content Indexed")

# 4. Chat UI
st.title("💬 GenAI Financial Agent")
st.caption("Ask questions about your documents or request visualizations (e.g., 'Show me a pie chart')")

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help with your finances today?"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please provide a Gemini API key in the sidebar.")
        else:
            try:
                model = genai.GenerativeModel(model_id)
                
                # Check for Chart Intent (Logic Routing)
                if any(x in prompt.lower() for x in ["pie", "chart", "graph", "plot", "visualize"]):
                    if csv_file:
                        df = pd.read_csv(csv_file)
                        num_cols = df.select_dtypes(include=['number']).columns.tolist()
                        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                        
                        if "pie" in prompt.lower() and cat_cols and num_cols:
                            fig = px.pie(df, names=cat_cols[0], values=num_cols[0], title="Financial Distribution")
                            st.plotly_chart(fig)
                            response_text = "I've generated the Pie Chart based on your CSV data."
                        elif num_cols:
                            fig = px.line(df, y=num_cols[0], title="Financial Trend Analysis")
                            st.plotly_chart(fig)
                            response_text = "Here is the trend visualization for your data."
                        else:
                            response_text = "I found your CSV, but there aren't enough numeric/categorical columns to plot."
                    else:
                        response_text = "I can't generate a chart without a CSV file. Please upload one in the sidebar."
                
                # Standard Text Response (RAG)
                else:
                    context = f"PDF Context: {st.session_state.pdf_context}\n"
                    if csv_file:
                        df_preview = pd.read_csv(csv_file).describe().to_string()
                        context += f"CSV Statistics: {df_preview}"
                    
                    full_prompt = f"System: Use context below to answer accurately.\n{context}\n\nUser: {prompt}"
                    response = model.generate_content(full_prompt)
                    response_text = response.text

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"Error: {str(e)}")
