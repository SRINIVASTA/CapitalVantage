import streamlit as st
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import plotly.express as px

# 1. Setup
st.set_page_config(page_title="GenAI Financial Agent", layout="wide")
st.title("🚀 Dynamic GenAI Financial Agent")

# Sidebar
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.divider()
    st.header("🤖 Model Selection")
    
    available_models = ["gemini-1.5-flash"] # Default fallback
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # DYNAMICALLY FETCH MODELS: This gets every model your key can access
            model_list = [
                m.name.replace('models/', '') 
                for m in genai.list_models() 
                if 'generateContent' in m.supported_generation_methods
            ]
            if model_list:
                available_models = model_list
        except Exception as e:
            st.error("Could not fetch models. Using default.")

    model_id = st.selectbox("Select an active model", available_models)

# 2. Logic Functions
def get_pdf_text(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        content = page.extract_text()
        if content: text += content
    return text

def agent_query(prompt, context=""):
    try:
        # Initializing the selected model
        model = genai.GenerativeModel(model_id)
        full_content = f"Context: {context[:20000]}\n\nRequest: {prompt}"
        response = model.generate_content(full_content)
        return response.text
    except Exception as e:
        return f"⚠️ Error with model {model_id}: {str(e)}"

# 3. Main Interface
if not api_key:
    st.info("Enter your API key in the sidebar to load available models.")
else:
    tab1, tab2 = st.tabs(["📄 PDF Summarizer", "📈 CSV Data Agent"])

    with tab1:
        pdf_file = st.file_uploader("Upload Financial PDF", type="pdf")
        if pdf_file and st.button("Summarize"):
            with st.spinner(f"Using {model_id}..."):
                text = get_pdf_text(pdf_file)
                res = agent_query("Summarize this financial document.", context=text)
                st.markdown(res)

    with tab2:
        csv_file = st.file_uploader("Upload Financial Data", type="csv")
        if csv_file:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head())
            query = st.text_input("Ask about this data:")
            if query:
                if any(x in query.lower() for x in ["plot", "chart", "visualize"]):
                    st.plotly_chart(px.line(df, title="Financial Trends"))
                else:
                    with st.spinner(f"Calculating with {model_id}..."):
                        stats = df.describe().to_string()
                        ans = agent_query(query, context=f"Data Stats:\n{stats}")
                        st.info(ans)
