import streamlit as st

from utils.text_preprocessor import preprocess_text, remove_stopwords
from utils.tfidf_vectorizer import compute_tf, compute_idf, compute_tfidf
from utils.similarity import cosine_similarity
from utils.skill_extractor import extract_skills

st.title("Smart Resume Analyzer (ATS)")

# Upload resume
uploaded_file = st.file_uploader("Upload Resume (TXT only)", type=["txt"])

# Job description input
job_description = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if uploaded_file is not None and job_description:
        
        # Read resume
        resume_text = uploaded_file.read().decode("utf-8")
        
        # Clean text
        resume_clean = remove_stopwords(preprocess_text(resume_text))
        jd_clean = remove_stopwords(preprocess_text(job_description))
        
        # TF-IDF
        documents = [resume_clean, jd_clean]
        
        tf_resume = compute_tf(resume_clean)
        tf_jd = compute_tf(jd_clean)
        
        idf = compute_idf(documents)
        
        tfidf_resume = compute_tfidf(tf_resume, idf)
        tfidf_jd = compute_tfidf(tf_jd, idf)
        
        # Similarity
        score = cosine_similarity(tfidf_resume, tfidf_jd)
        
        # Skills
        resume_skills = extract_skills(resume_clean)
        jd_skills = extract_skills(jd_clean)
        missing_skills = jd_skills - resume_skills
        
        # Output
        st.subheader(f"ATS Score: {round(score * 100, 2)}%")
        st.write("Missing Skills:", ", ".join(missing_skills))
    
    else:
        st.warning("Please upload resume and enter job description")