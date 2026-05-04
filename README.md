# 🚀 GenAI Financial Agent (Powered by Gemini)

**Developed by:** [Srinivasta](https://github.com)  
**Role:** Data Scientist

A professional Generative AI application built to showcase LLM orchestration, RAG pipelines, and Document Intelligence. This agent uses Google’s Gemini Pro and Flash models to analyze complex financial datasets and unstructured documents, providing intelligent summaries and real-time data visualizations via a Streamlit interface.

## 🌟 Key Features

*   **Chat-Based Interface:** A stateful, session-aware chat experience that remembers conversation history.
*   **Dual-Mode Intelligence:**
    *   **PDF Summarizer:** Extracts text from financial reports and provides executive summaries using a RAG-inspired context window.
    *   **Data Analyst Agent:** Processes CSV data to answer quantitative questions and perform statistical analysis.
*   **PDF-to-Data Conversion:** Advanced agentic logic that identifies tables within unstructured PDF text, converts them into structured CSV format via the LLM, and enables immediate data manipulation.
*   **Dynamic Visualizations:** Automatically generates interactive Pie Charts, Bar Graphs, and Line Trends using Plotly based on natural language intent.
*   **Dynamic Model Discovery:** Automatically fetches and lists all Gemini models available to your specific API key for seamless switching.

## 🛠️ Tech Stack

- **Model:** Google Gemini (1.5 Pro / 1.5 Flash / 2.0 Flash)
- **Frontend:** [Streamlit](https://streamlit.io)
- **Orchestration:** [Google GenAI SDK](https://github.com)
- **Data Processing:** Pandas, PyPDF2
- **Visualization:** Plotly Express

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.11 or 3.12 (Recommended)
*   A Google Gemini API Key. Obtain one at [Google AI Studio](https://google.com).

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/CapitalVantage.git
cd CapitalVantage

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App

```bash
streamlit run app.py
```

## 📂 Project Structure

```text
├── app.py              # Main Streamlit application and agentic logic
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation and portfolio highlights
```

## 🤖 Agentic Logic Overview

Unlike standard chatbots, this agent performs **Intent Routing**:

1.  **Detection:** It analyzes the user's prompt to detect if a "visualization" or "data structuring" task is needed.
2.  **Extraction:** If only a PDF is present, it triggers a sub-task to extract and structure unstructured text into a temporary DataFrame.
3.  **Execution:** It passes the structured data to Plotly to render interactive visuals, overcoming the text-only limitation of standard LLMs.

---
**Created by Srinivasta, Data Scientist.**  
*Leveraging AI to bridge the gap between unstructured data and financial insights.*
