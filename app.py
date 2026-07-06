import streamlit as st
import os
import io
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage

from docx import Document
from pypdf import PdfReader

# Page config
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 AI Technical Interview Coach")
st.markdown("*Powered by LangGraph + RAG*")

# Sidebar
st.sidebar.header("⚙️ Settings")
groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key",
    type="password",
    help="Get a free key at console.groq.com"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 How it works:")
st.sidebar.markdown("""
1. Upload your CV
2. Upload job description
3. Answer 5 questions
4. Get performance report!
""")

if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar!")
    st.info("🔑 Get a free key at [console.groq.com](https://console.groq.com)")
    st.stop()

# Setup LLM
llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

# State definition
class InterviewState(TypedDict):
    cv_text: str
    jd_text: str
    extracted_skills: str
    retrieved_context: str
    current_question: str
    user_answer: str
    score: int
    feedback: str
    hint: str
    difficulty: str
    question_count: int
    total_score: int
    asked_questions: list
    final_report: str
    messages: Annotated[list, add_messages]

# File reading function
def read_file(file_content, filename):
    try:
        if filename.endswith(".txt"):
            return file_content.decode("utf-8")
        elif filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_content))
            return "".join([page.extract_text() for page in reader.pages])
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(file_content))
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# Initialize session state
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "state" not in st.session_state:
    st.session_state.state = {}
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "scores" not in st.session_state:
    st.session_state.scores = []

# ============================================
# STAGE 1 - Upload files
# ============================================
if st.session_state.stage == "upload":
    st.header("📄 Step 1: Upload Your Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your CV")
        cv_file = st.file_uploader(
            "Upload CV",
            type=["pdf", "docx", "txt"],
            key="cv_upload"
        )
    
    with col2:
        st.subheader("Job Description")
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=["pdf", "docx", "txt"],
            key="jd_upload"
        )
    
    if cv_file and jd_file:
        st.success("✅ Both files uploaded!")
        
        if st.button("🚀 Start Interview Preparation", use_container_width=True):
            with st.spinner("📊 Analyzing documents and setting up RAG..."):
                try:
                    # Read files
                    cv_text = read_file(cv_file.read(), cv_file.name)
                    jd_text = read_file(jd_file.read(), jd_file.name)
                    
                    # Setup RAG
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    cv_chunks = text_splitter.create_documents(
                        [cv_text], metadatas=[{"source": "cv"}]
                    )
                    jd_chunks = text_splitter.create_documents(
                        [jd_text], metadatas=[{"source": "jd"}]
                    )
                    all_chunks = cv_chunks + jd_chunks
                    
                    embeddings = SentenceTransformerEmbeddings(
                        model_name="all-MiniLM-L6-v2"
                    )
                    vectorstore = Chroma.from_documents(
                        documents=all_chunks,
                        embedding=embeddings
                    )
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    
                    # Analyze resume
                    prompt = f"""
                    You are a senior recruiter.
                    Analyze this CV and Job Description:
                    CV: {cv_text}
                    JD: {jd_text}
                    Extract key skills and identify gaps.
                    """
                    skills = llm.invoke(prompt).content
                    
                    # Initialize state
                    st.session_state.state = {
                        "cv_text": cv_text,
                        "jd_text": jd_text,
                        "extracted_skills": skills,
                        "retrieved_context": "",
                        "current_question": "",
                        "user_answer": "",
                        "score": 0,
                        "feedback": "",
                        "hint": "",
                        "difficulty": "medium",
                        "question_count": 0,
                        "total_score": 0,
                        "asked_questions": [],
                        "final_report": "",
                        "messages": []
                    }
                    st.session_state.scores = []
                    st.session_state.stage = "interview"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================
# STAGE 2 - Interview
# ============================================
elif st.session_state.stage == "interview":
    state = st.session_state.state
    
    # Progress bar
    progress = state['question_count'] / 5
    st.progress(progress)
    st.markdown(f"**Question {state['question_count'] + 1} of 5** | Difficulty: `{state['difficulty']}`")
    
    # Score tracker
    if st.session_state.scores:
        cols = st.columns(len(st.session_state.scores))
        for i, s in enumerate(st.session_state.scores):
            with cols[i]:
                color = "🟢" if s >= 7 else "🟡" if s >= 5 else "🔴"
                st.metric(f"Q{i+1}", f"{color} {s}/10")
    
    st.markdown("---")
    
    # Generate question if needed
    if not state.get('current_question') or state.get('need_new_question', True):
        with st.spinner("🤔 Generating your question..."):
            try:
                retriever = st.session_state.retriever
                docs = retriever.invoke(
                    f"Interview questions for: {state['extracted_skills']}"
                )
                context = "\n\n".join([d.page_content for d in docs])
                
                prompt = f"""
                You are a senior technical interviewer.
                Generate ONE interview question based on:
                Context: {context}
                Skills: {state['extracted_skills']}
                Difficulty: {state['difficulty']}
                Already asked: {state['asked_questions']}
                
                Rules:
                - easy = beginner level
                - medium = intermediate level
                - hard = advanced level
                - ONE question only
                - Don't repeat previous questions
                - Return ONLY the question
                """
                question = llm.invoke(prompt).content
                state['current_question'] = question
                state['retrieved_context'] = context
                state['need_new_question'] = False
                st.session_state.state = state
                
            except Exception as e:
                st.error(f"Error generating question: {e}")
    
    # Show question
    st.subheader("🎤 Interview Question:")
    st.info(state['current_question'])
    
    # Answer input
    answer = st.text_area(
        "Your Answer:",
        placeholder="Type your answer here...",
        height=150,
        key=f"answer_{state['question_count']}"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("✅ Submit Answer", use_container_width=True)
    with col2:
        skip = st.button("⏭️ Skip", use_container_width=True)
    
    if submit and answer:
        with st.spinner("📊 Evaluating your answer..."):
            try:
                eval_prompt = f"""
                Question: {state['current_question']}
                Answer: {answer}
                Job requirements: {state['jd_text']}
                
                Evaluate and reply EXACTLY:
                SCORE: [1-10]
                FEEDBACK: [detailed feedback]
                """
                evaluation = llm.invoke(eval_prompt).content
                
                score = 5
                feedback = ""
                for line in evaluation.split("\n"):
                    if "SCORE:" in line:
                        try:
                            score = int(''.join(
                                filter(str.isdigit, line.split("SCORE:")[1])
                            ))
                        except:
                            score = 5
                    if "FEEDBACK:" in line:
                        feedback = line.split("FEEDBACK:")[1].strip()
                
                # Update difficulty
                if score >= 8:
                    difficulty = "hard"
                elif score >= 5:
                    difficulty = "medium"
                else:
                    difficulty = "easy"
                
                # Show result
                if score >= 7:
                    st.success(f"✅ Score: {score}/10")
                elif score >= 5:
                    st.warning(f"⚠️ Score: {score}/10")
                else:
                    st.error(f"❌ Score: {score}/10")
                
                st.write(f"**Feedback:** {feedback}")
                
                # Generate hint if low score
                if score < 5:
                    with st.spinner("💡 Generating hint..."):
                        hint_prompt = f"""
                        Give a helpful hint for this question:
                        {state['current_question']}
                        Don't reveal the answer.
                        """
                        hint = llm.invoke(hint_prompt).content
                        st.info(f"💡 **Hint:** {hint}")
                
                # Update state
                st.session_state.scores.append(score)
                state['question_count'] += 1
                state['total_score'] += score
                state['asked_questions'].append(state['current_question'])
                state['difficulty'] = difficulty
                state['need_new_question'] = True
                st.session_state.state = state
                
                if state['question_count'] >= 5:
                    st.session_state.stage = "report"
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error evaluating: {e}")
    
    if skip:
        state['question_count'] += 1
        state['asked_questions'].append(state['current_question'])
        state['need_new_question'] = True
        st.session_state.scores.append(0)
        st.session_state.state = state
        
        if state['question_count'] >= 5:
            st.session_state.stage = "report"
        st.rerun()

# ============================================
# STAGE 3 - Final Report
# ============================================
elif st.session_state.stage == "report":
    state = st.session_state.state
    
    st.header("📊 Your Interview Performance Report")
    
    # Score summary
    total = state['total_score']
    maximum = state['question_count'] * 10
    percentage = (total / maximum * 100) if maximum > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Score", f"{total}/{maximum}")
    with col2:
        st.metric("Percentage", f"{percentage:.1f}%")
    with col3:
        if percentage >= 70:
            st.metric("Result", "✅ Pass")
        else:
            st.metric("Result", "❌ Needs Work")
    
    # Score progression
    if st.session_state.scores:
        st.subheader("📈 Score Progression")
        cols = st.columns(len(st.session_state.scores))
        for i, s in enumerate(st.session_state.scores):
            with cols[i]:
                color = "🟢" if s >= 7 else "🟡" if s >= 5 else "🔴"
                st.metric(f"Q{i+1}", f"{color} {s}/10")
    
    st.markdown("---")
    
    # Generate report
    with st.spinner("📝 Generating your detailed report..."):
        try:
            report_prompt = f"""
            You are a senior HR manager.
            Generate a professional interview performance report:
            
            Candidate Skills: {state['extracted_skills']}
            Total Score: {state['total_score']} out of {state['question_count'] * 10}
            Questions Asked: {state['asked_questions']}
            Last Feedback: {state['feedback']}
            Job Requirements: {state['jd_text']}
            
            Include:
            1. 📊 Overall performance summary
            2. ✅ Strong areas
            3. ❌ Weak areas
            4. 📚 Topics to study
            5. 🎯 Final recommendation
            """
            report = llm.invoke(report_prompt).content
            st.markdown(report)
            
        except Exception as e:
            st.error(f"Error generating report: {e}")
    
    st.markdown("---")
    
    if st.button("🔄 Start New Interview", use_container_width=True):
        st.session_state.stage = "upload"
        st.session_state.state = {}
        st.session_state.scores = []
        st.rerun()
