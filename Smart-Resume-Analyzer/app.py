import streamlit as st
from utils.deep_learning_match import semantic_similarity
import re
import pandas as pd

from utils.text_extractor import extract_text
from utils.text_preprocessor import preprocess_text, remove_stopwords
from utils.tfidf_vectorizer import compute_tf, compute_idf, compute_tfidf
from utils.similarity import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart ATS Resume Analyzer",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "results" not in st.session_state:
    st.session_state.results = []

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.skill-box{
    padding:6px 12px;
    border-radius:8px;
    margin:4px;
    display:inline-block;
    font-size:13px;
    font-weight:500;
}

.match{
    background:#1f6f4a;
    color:white;
}

.missing{
    background:#7a1f2b;
    color:white;
}

.neutral{
    background:#2c2f36;
    color:#ddd;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SKILLS ----------------
skills_list = [

    "python",
    "java",
    "react",
    "sql",
    "mongodb",
    "firebase",
    "nodejs",

    "machine learning",
    "data structures",
    "system design",
    "aws",

    "cyber security",
    "ethical hacking",
    "penetration testing",
    "network security",
    "linux",
    "siem",
    "firewall",
    "incident response",
    "cloud security",
    "risk assessment",
    "vulnerability assessment"
]

# ---------------- FUNCTIONS ----------------
def extract_skills(text):

    return [
        skill for skill in skills_list
        if skill in text.lower()
    ]

def keyword_density(text):

    text = text.lower()

    return {
        skill: text.count(skill)
        for skill in skills_list
    }

def check_sections(text):

    text = text.lower()

    sections = [
        "skills",
        "project",
        "experience",
        "education"
    ]

    missing = []

    for sec in sections:

        if sec not in text:
            missing.append(f"{sec.capitalize()} missing")

    return missing

def detect_weak_words(text):

    weak_words = [
        "learned",
        "worked on",
        "done",
        "made"
    ]

    return [
        w for w in weak_words
        if w in text.lower()
    ]

def check_numbers(text):

    return len(re.findall(r"\d+", text))

def analyze_bullets(text):

    lines = text.split("\n")

    feedback = []

    action_words = [
        "developed",
        "designed",
        "implemented",
        "built",
        "optimized"
    ]

    for line in lines:

        if len(line.strip()) > 5:

            issues = []

            if not any(w in line.lower() for w in action_words):
                issues.append("Add action verb")

            if not re.search(r"\d+", line):
                issues.append("Add measurable result")

            if issues:
                feedback.append((line, issues))

    return feedback

# ---------------- HEADER ----------------
st.title("Smart ATS Resume Analyzer")

st.caption(
    "Upload multiple resumes and compare them using AI + Deep Learning"
)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:

    resume_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

with col2:

    jd_text = st.text_area(
        "Paste Job Description",
        height=250
    )

# ---------------- ANALYZE BUTTON ----------------
analyze = st.button("Analyze Resumes")

if analyze:

    if resume_files and jd_text:

        results = []

        for resume_file in resume_files:

            resume_text = extract_text(resume_file)

            resume_clean = remove_stopwords(
                preprocess_text(resume_text)
            )

            jd_clean = remove_stopwords(
                preprocess_text(jd_text)
            )

            # ---------------- SKILL MATCH ----------------
            resume_skills = set(extract_skills(resume_clean))
            jd_skills = set(extract_skills(jd_clean))

            match = resume_skills & jd_skills
            missing = jd_skills - resume_skills

            # ---------------- TF-IDF ----------------
            docs = [resume_clean, jd_clean]

            tf1 = compute_tf(resume_clean)
            tf2 = compute_tf(jd_clean)

            idf = compute_idf(docs)

            tfidf1 = compute_tfidf(tf1, idf)
            tfidf2 = compute_tfidf(tf2, idf)

            cosine_score = cosine_similarity(
                tfidf1,
                tfidf2
            )

            # ---------------- DEEP LEARNING ----------------
            deep_score = semantic_similarity(
                resume_text,
                jd_text
            )

            # ---------------- FINAL SCORE ----------------
            skill_score = (
                len(match) / len(jd_skills)
                if jd_skills else 0
            )

            final_score = (
                0.2 * skill_score +
                0.2 * cosine_score +
                0.6 * deep_score
            )

            results.append({

                "Resume": resume_file.name,

                "ATS Score": round(final_score * 100, 2),

                "Skill Match": round(skill_score * 100, 2),

                "Similarity": round(cosine_score * 100, 2),

                "AI Match": round(deep_score * 100, 2),

                "match": match,

                "missing": missing,

                "text": resume_text,

                "file": resume_file
            })

        # ---------------- SORT ----------------
        results = sorted(
            results,
            key=lambda x: x["ATS Score"],
            reverse=True
        )

        # ---------------- ADD RANK ----------------
        for i, r in enumerate(results):

            r["Rank"] = i + 1

        st.session_state.results = results

    else:

        st.warning(
            "Upload resumes and paste job description"
        )

# ---------------- SHOW RESULTS ----------------
if st.session_state.results:

    results = st.session_state.results

    # ---------------- TABLE ----------------
    st.subheader("Resume Ranking")

    df = pd.DataFrame(results)[[
        "Rank",
        "Resume",
        "ATS Score",
        "Skill Match",
        "Similarity",
        "AI Match"
    ]]

    st.dataframe(
        df,
        use_container_width=True
    )

    # ---------------- GRAPH ----------------
    st.subheader("Resume Ranking Graph")

    st.bar_chart(
        df.set_index("Resume")["ATS Score"]
    )

    # ---------------- SCORE BREAKDOWN ----------------
    st.subheader("Top Resume Score Breakdown")

    top = results[0]

    breakdown_df = pd.DataFrame({

        "Metric": [
            "Skill Match",
            "TF-IDF Similarity",
            "AI Match"
        ],

        "Score": [
            top["Skill Match"],
            top["Similarity"],
            top["AI Match"]
        ]
    })

    st.bar_chart(
        breakdown_df.set_index("Metric")
    )

    # ---------------- RESUME COMPARISON ----------------
    st.subheader("Resume vs Resume Comparison")

    names = [r["Resume"] for r in results]

    col1, col2 = st.columns(2)

    with col1:

        r1 = st.selectbox(
            "Select Resume 1",
            names,
            key="resume1"
        )

    with col2:

        r2 = st.selectbox(
            "Select Resume 2",
            names,
            key="resume2"
        )

    if r1 and r2:

        res1 = next(
            r for r in results
            if r["Resume"] == r1
        )

        res2 = next(
            r for r in results
            if r["Resume"] == r2
        )

        compare_df = pd.DataFrame({

            "Metric": [
                "ATS Score",
                "Skill Match",
                "Similarity",
                "AI Match"
            ],

            r1: [
                res1["ATS Score"],
                res1["Skill Match"],
                res1["Similarity"],
                res1["AI Match"]
            ],

            r2: [
                res2["ATS Score"],
                res2["Skill Match"],
                res2["Similarity"],
                res2["AI Match"]
            ]
        })

        st.dataframe(
            compare_df,
            use_container_width=True
        )

    # ---------------- DOWNLOAD ----------------
    st.subheader("Download Best Resume")

    best_resume = results[0]

    st.download_button(
        label="Download Top Ranked Resume",
        data=best_resume["file"],
        file_name=best_resume["Resume"]
    )

    # ---------------- DETAILS ----------------
    st.subheader("Detailed Resume Analysis")

    for r in results:

        with st.expander(
            f'{r["Resume"]} | Rank #{r["Rank"]}'
        ):

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "ATS Score",
                f'{r["ATS Score"]}%'
            )

            col2.metric(
                "Skill Match",
                f'{r["Skill Match"]}%'
            )

            col3.metric(
                "Similarity",
                f'{r["Similarity"]}%'
            )

            col4.metric(
                "AI Match",
                f'{r["AI Match"]}%'
            )

            # ---------------- MATCHING ----------------
            st.markdown("### Matching Skills")

            if r["match"]:

                st.markdown(
                    " ".join([
                        f'<span class="skill-box match">{s}</span>'
                        for s in r["match"]
                    ]),
                    unsafe_allow_html=True
                )

            else:
                st.write("No matching skills found")

            # ---------------- MISSING ----------------
            st.markdown("### Missing Skills")

            if r["missing"]:

                st.markdown(
                    " ".join([
                        f'<span class="skill-box missing">{s}</span>'
                        for s in r["missing"]
                    ]),
                    unsafe_allow_html=True
                )

            else:
                st.write("No missing skills")

            # ---------------- KEYWORD ----------------
            st.markdown("### Keyword Usage")

            density = keyword_density(r["text"])

            st.markdown(
                " ".join([
                    f'<span class="skill-box neutral">{k}: {v}</span>'
                    for k, v in density.items()
                ]),
                unsafe_allow_html=True
            )

            # ---------------- SECTION CHECK ----------------
            st.markdown("### Section Analysis")

            issues = check_sections(r["text"])

            if issues:

                st.write(issues)

            else:
                st.success(
                    "All important sections present"
                )

            # ---------------- WEAK WORDS ----------------
            st.markdown("### Weak Words")

            weak = detect_weak_words(r["text"])

            st.write(weak if weak else "None")

            # ---------------- IMPACT ----------------
            st.markdown("### Impact Analysis")

            num_count = check_numbers(r["text"])

            st.write(
                f"Numbers used in resume: {num_count}"
            )

            # ---------------- BULLET FEEDBACK ----------------
            st.markdown("### Bullet Feedback")

            feedback = analyze_bullets(r["text"])

            if feedback:

                for line, issues in feedback[:5]:

                    st.write(line)

                    st.caption(
                        ", ".join(issues)
                    )

            else:
                st.write(
                    "Good bullet structure detected"
                )