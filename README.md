# 🎯 Technical Interview Coach AI

An AI-powered technical interview preparation system that analyzes your CV and job description to conduct personalized adaptive mock interviews.

## 🚀 Live Demo
[Try it here](your-streamlit-link-here) ← update after deployment

## 📋 What it does

Upload your CV and job description → get a personalized mock interview → receive detailed performance report!
Upload CV + Job Description
↓
Node 1 → Analyzes your resume and extracts skills
↓
Node 2 → Searches relevant context using RAG
↓
Node 3 → Generates personalized interview question
↓
Node 4 → Evaluates your answer + adjusts difficulty
↓
After 5 questions
↓
Node 5 → Generates detailed performance report
## ✨ Features

- 📄 **Smart Resume Analysis** — Extracts skills and identifies gaps vs job requirements
- 🔍 **RAG-Powered Questions** — Questions based on YOUR actual CV and job description
- 🎯 **Adaptive Difficulty** — Questions get harder or easier based on your performance
- 💡 **Real-time Hints** — Helpful hints when answers need improvement
- 📊 **Score Tracking** — Tracks performance across all questions
- 🔄 **No Repeated Questions** — Memory ensures fresh questions every time
- ✅ **Error Handling** — Graceful error recovery in all nodes
- 📋 **Detailed Report** — Final report with strengths, weaknesses and study recommendations

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **LangGraph** | Multi-node agent orchestration |
| **RAG** | Retrieval Augmented Generation |
| **ChromaDB** | Vector database for document storage |
| **HuggingFace Embeddings** | Text to vector conversion |
| **Groq LLM API** | Fast free LLM inference (Llama 3.3 70B) |
| **Streamlit** | Web interface and deployment |
| **Python** | Core development |
| **PyPDF/python-docx** | Document parsing |

## 🏗️ Architecture

### Graph Structure:
analyze_resume → retrieve_knowledge → generate_question → evaluate_answer
↑↓
←←←←←← loop back (if < 5 questions) ←←←←
↓
generate_report
### 5 Specialized Nodes:
1. **Resume Analyzer** — Reads CV + JD, extracts skills and gaps
2. **RAG Retriever** — Searches ChromaDB for relevant context
3. **Question Generator** — Creates adaptive questions based on difficulty
4. **Answer Evaluator** — Scores answers, adjusts difficulty, generates hints
5. **Report Generator** — Creates comprehensive performance report

## 🚀 How to Run

### Option 1 — Use the live app:
Click the live demo link above!

### Option 2 — Run locally in Google Colab:
1. Open `InterviewCopilot_V2.ipynb` in Google Colab
2. Add your Groq API key to Colab secrets as `GroqKey`
3. Run all cells in order
4. Upload your CV and job description files
5. Start your mock interview!

### Option 3 — Run Streamlit locally:
```bash
git clone https://github.com/sonaina005/technical-interview-coach-ai
cd technical-interview-coach-ai
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
technical-interview-coach-ai/
│
├── InterviewCopilot_V2.ipynb  ← Main notebook
├── app.py                      ← Streamlit web app
├── requirements.txt            ← Dependencies
└── README.md                   ← You are here!
