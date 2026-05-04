# 🚀 GenAI Financial Agent (Powered by Gemini)

A Generative AI application built to showcase LLM orchestration, RAG pipelines, and document intelligence. This agent uses Google’s Gemini Pro/Flash models to analyze financial datasets and unstructured documents. It provides intelligent summaries and real-time data visualizations via a Streamlit interface.

## 🌟 Key Features

*   Chat-Based Interface: A stateful chat experience that remembers conversation history.
*   Dual-Mode Intelligence:
    *   PDF Summarizer: Extracts text from financial reports (PDF) and provides executive summaries using a RAG-inspired context window.
    *   Data Analyst Agent: Processes CSV data to answer quantitative questions.
*   PDF-to-Data Conversion: Agentic logic that can identify tables within a PDF, convert them into structured CSV format via the LLM, and render charts.
*   Dynamic Visualizations: Automatically generates Pie Charts, Bar Graphs, and Line Trends using Plotly based on natural language intent.
*   Dynamic Model Discovery: Automatically fetches and lists all Gemini models available to your specific API key.

## 🛠️ Tech Stack

- Model: Google Gemini (1.5 Pro / 1.5 Flash)
- Frontend: [Streamlit](https://streamlit.io)
- Orchestration: [Google GenAI SDK](https://github.com)
- Data Processing: Pandas, PyPDF2
- Visualization: Plotly Express

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.11 or 3.12 (Recommended)
*   A Google Gemini API Key. Get one at [Google AI Studio](https://google.com).

### 2. Installation

```bash
# Clone the repository
git clone https://github.com
cd genai-financial-agent

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App

```bash
streamlit run app.py
```

## 📂 Project Structure

```text
├── app.py              # Main Streamlit application logic
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🤖 Agentic Logic Overview

Unlike standard chatbots, this agent performs Intent Routing:

1.  It analyzes the user's prompt to detect if a "visualization" is needed.
2.  If only a PDF is present, it triggers a data extraction sub-task to structure unstructured text into a temporary DataFrame.
3.  It passes the structured data to Plotly to render interactive visuals, overcoming the text-only limitation of standard LLMs.

## 📄 License

Distributed under the MIT License.
Use code with caution.
