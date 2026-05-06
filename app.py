import streamlit as st
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import plotly.express as px
import io
from PIL import Image

# --- 1. CONFIGURATION & SESSION STATE ---
st.set_page_config(page_title="GenAI Multi-Asset Agent", page_icon="💰", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_df" not in st.session_state:
    st.session_state.extracted_df = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# --- 2. SIDEBAR: SETTINGS & UPLOADS ---
with st.sidebar:
    st.title("⚙️ Agent Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    available_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            fetched_models = [m.name.replace('models/', '') for m in genai.list_models() 
                             if 'generateContent' in m.supported_generation_methods]
            if fetched_models: available_models = fetched_models
        except:
            st.warning("Using default model list.")

    model_id = st.selectbox("Select Active Model", available_models)
    
    st.divider()
    st.header("📁 Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload Financial Data (PDF, CSV, XLSX, TXT, Image)", 
        type=["pdf", "csv", "xlsx", "xls", "txt", "png", "jpg", "jpeg"]
    )

    if st.button("Clear Chat & Memory"):
        st.session_state.messages = []
        st.session_state.extracted_df = None
        st.session_state.suggestions = []
        st.rerun()

# --- 3. UNIVERSAL DATA PROCESSING ---
context_text = ""
context_image = None

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_ext == "pdf":
            reader = PdfReader(uploaded_file)
            context_text = "".join([page.extract_text() or "" for page in reader.pages])
            st.sidebar.success("PDF Indexed")
            
        elif file_ext in ["csv"]:
            st.session_state.extracted_df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV Loaded")
            
        elif file_ext in ["xlsx", "xls"]:
            st.session_state.extracted_df = pd.read_excel(uploaded_file)
            st.sidebar.success("Excel Loaded")
            
        elif file_ext == "txt":
            context_text = uploaded_file.read().decode("utf-8")
            st.sidebar.success("Text File Indexed")
            
        elif file_ext in ["png", "jpg", "jpeg"]:
            context_image = Image.open(uploaded_file)
            st.sidebar.image(context_image, caption="Uploaded Image", use_column_width=True)
            st.sidebar.success("Image Loaded")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# --- 4. PROACTIVE SUGGESTIONS ---
if (context_text or st.session_state.extracted_df is not None or context_image) and api_key and not st.session_state.suggestions:
    with st.spinner("Analyzing context..."):
        model = genai.GenerativeModel(model_id)
        # Create a tiny preview for the suggestion engine
        preview = context_text[:1000] if context_text else "Tabular Data" if st.session_state.extracted_df is not None else "Image Content"
        s_prompt = f"Suggest 3 short financial questions based on this: {preview}. Return ONLY the questions."
        try:
            res = model.generate_content(s_prompt)
            st.session_state.suggestions = [q.strip() for q in res.text.split('\n') if q.strip()][:3]
        except:
            st.session_state.suggestions = ["Summarize this data", "Perform Risk Analysis", "Key takeaways"]

if st.session_state.suggestions:
    st.sidebar.subheader("💡 Suggestions")
    for q in st.session_state.suggestions:
        if st.sidebar.button(q):
            st.session_state.active_prompt = q

# --- 5. MAIN INTERFACE & CHAT ---
st.title("💬 GenAI Financial Agent")
st.info(f"Mode: **Multimodal Analysis** | Model: **{model_id}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask about your financial files...")
if st.session_state.get("active_prompt"):
    user_query = st.session_state.active_prompt
    del st.session_state.active_prompt

if user_query:
    if not api_key:
        st.error("🔑 API Key required.")
    elif not uploaded_file:
        st.warning("⚠️ Please upload a file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                model = genai.GenerativeModel(model_id)
                
                # Logic for Visualizations
                if any(k in user_query.lower() for k in ["chart", "plot", "graph"]):
                    if st.session_state.extracted_df is not None:
                        df = st.session_state.extracted_df
                        num_cols = df.select_dtypes(include=['number']).columns.tolist()
                        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                        fig = px.line(df, x=cat_cols[0] if cat_cols else None, y=num_cols[0] if num_cols else None)
                        st.plotly_chart(fig, use_container_width=True)
                        ans = "Here is the visualization based on the tabular data."
                    else:
                        ans = "I need a CSV or Excel file to create a chart."
                
                # Multimodal Chat Logic
                else:
                    content_payload = []
                    if context_text: content_payload.append(f"Document Text: {context_text[:15000]}")
                    if st.session_state.extracted_df is not None: content_payload.append(f"Data Sample: {st.session_state.extracted_df.head(10).to_string()}")
                    if context_image: content_payload.append(context_image)
                    content_payload.append(f"User Question: {user_query}")
                    
                    response = model.generate_content(content_payload)
                    ans = response.text

                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})

            except Exception as e:
                st.error(f"Error: {e}")
