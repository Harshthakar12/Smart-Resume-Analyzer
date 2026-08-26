import streamlit as st
import os
import re
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from google import genai

from utils.text_extractor import extract_text
from utils.text_preprocessor import preprocess_text, remove_stopwords
from utils.tfidf_vectorizer import compute_tf, compute_idf, compute_tfidf
from utils.similarity import cosine_similarity, cot_llm_ats_score
from utils.skill_extractor import extract_skills_with_llm

st.set_page_config(page_title="Prompt Engineering Resume Analyzer", layout="wide")

# Sidebar - API Key Configuration
st.sidebar.title("⚙️ Setup")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

dark_mode = st.sidebar.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)

# Action verbs helper
action_verbs = ["developed", "implemented", "designed", "built", "optimized", "created", "engineered", "improved"]

def detect_weak_words(text):
    weak_words = ["learned", "worked on", "done", "made"]
    return [w for w in weak_words if w in text.lower()]

def check_numbers(text):
    return len(re.findall(r"\d+", text))

def analyze_bullets(text):
    lines = text.split("\n")
    results = []
    for line in lines:
        line = line.strip()
        if len(line) < 20:
            continue
        issues = []
        if not any(v in line.lower() for v in action_verbs):
            issues.append("Use action verb")
        if not re.search(r"\d+", line):
            issues.append("Add numbers")
        if any(w in line.lower() for w in ["worked on", "made", "done"]):
            issues.append("Avoid weak words")
        results.append((line, issues if issues else ["Strong"]))
    return results

def generate_persona_feedback(resume_text, jd_text, api_key):
    """Persona / Role Prompting Technique."""
    if not api_key:
        return "Please supply a Gemini API Key in the sidebar to view Persona-based AI feedback."
    
    client = genai.Client(api_key=api_key)
    prompt = f"""You are a Senior Technical Recruiter and Hiring Manager at a top-tier tech company. 
Review this resume for the candidate against the provided job description. Give direct, high-impact advice on how the candidate can optimize their resume bullet points, tone, and positioning to get hired.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

CANDIDATE RESUME:
\"\"\"{resume_text}\"\"\"

Provide concise, bulleted, actionable feedback:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error building suggestions: {e}"

def generate_pdf(score, match, missing):
    doc = SimpleDocTemplate("resume_report.pdf")
    styles = getSampleStyleSheet()
    content = [
        Paragraph(f"ATS Score: {score}%", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"Matching Skills: {', '.join(match)}", styles["Normal"]),
        Paragraph(f"Missing Skills: {', '.join(missing)}", styles["Normal"])
    ]
    doc.build(content)
    return "resume_report.pdf"

# Main UI
st.title("⚡ AI-Powered Smart Resume Analyzer")
st.caption("Enhanced using Few-Shot, Chain-of-Thought, and Persona Prompt Engineering Techniques")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])
with col2:
    jd_text = st.text_area("Paste Job Description", height=150)

if st.button("🚀 Run Enhanced Analysis"):
    if resume_file and jd_text:
        resume_text = extract_text(resume_file)
        
        # 1. Few-Shot Skill Extraction
        st.subheader("🛠️ Prompt-Engineered Skill Extraction (Few-Shot)")
        resume_skills = set(extract_skills_with_llm(resume_text, gemini_api_key))
        jd_skills = set(extract_skills_with_llm(jd_text, gemini_api_key))
        
        match = resume_skills & jd_skills
        missing = jd_skills - resume_skills
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.success(f"**Matching Skills ({len(match)}):** {', '.join(match) if match else 'None'}")
        with m_col2:
            st.error(f"**Missing Skills ({len(missing)}):** {', '.join(missing) if missing else 'None'}")

        st.markdown("---")
        
        # 2. Chain-of-Thought Score & Analysis
        st.subheader("🧠 Chain-of-Thought (CoT) ATS Evaluation")
        cot_result = cot_llm_ats_score(resume_text, jd_text, gemini_api_key)
        
        st.metric(label="Overall Prompt-Engineered ATS Score", value=f"{cot_result.get('ats_score', 0)}%")
        
        with st.expander("🔍 View AI Step-by-Step Chain-of-Thought Reasoning", expanded=True):
            st.info(cot_result.get("thought_process", "No rationale generated."))

        # Standard TF-IDF comparison baseline
        resume_clean = remove_stopwords(preprocess_text(resume_text))
        jd_clean = remove_stopwords(preprocess_text(jd_text))
        docs = [resume_clean, jd_clean]
        tf1, tf2 = compute_tf(resume_clean), compute_tf(jd_clean)
        idf = compute_idf(docs)
        tfidf1, tfidf2 = compute_tfidf(tf1, idf), compute_tfidf(tf2, idf)
        legacy_cosine = round(cosine_similarity(tfidf1, tfidf2) * 100, 2)
        
        st.caption(f"📊 *Baseline TF-IDF Cosine Similarity Score: {legacy_cosine}%*")

        st.markdown("---")

        # 3. Persona-Based Suggestions
        st.subheader("👔 Senior Recruiter Persona Feedback")
        recruiter_feedback = generate_persona_feedback(resume_text, jd_text, gemini_api_key)
        st.markdown(recruiter_feedback)

        st.markdown("---")

        # Bullet Point Analysis
        st.subheader("📍 Detailed Bullet Line Analysis")
        bullets = analyze_bullets(resume_text)
        for line, issues in bullets:
            st.write(f"👉 {line}")
            for issue in issues:
                if issue == "Strong":
                    st.success(issue)
                else:
                    st.warning(issue)

        # PDF Download Button
        pdf_path = generate_pdf(cot_result.get("ats_score", 0), match, missing)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Summary", f, file_name="resume_analysis.pdf")
            
    else:
        st.warning("Please upload a resume file and paste a Job Description to begin.")