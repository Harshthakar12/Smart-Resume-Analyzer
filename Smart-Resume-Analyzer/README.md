# 📄 Smart Resume Analyzer (ATS Score Checker)

## 🚀 Overview

Smart Resume Analyzer is an AI-based project that evaluates a resume against a job description and calculates an **ATS (Applicant Tracking System) score**.

It helps users understand how well their resume matches a job and identifies **missing skills**.

---

## 🧠 Features

* 📄 Upload Resume (PDF/TXT)
* 🧹 Text Preprocessing (cleaning + stopwords removal)
* 📊 TF-IDF based similarity scoring
* 📈 Cosine Similarity for ATS Score
* ✅ Matched Skills Detection
* ❌ Missing Skills Identification
* 🌐 Interactive UI using Streamlit

---

## 🛠️ Tech Stack

* Python
* Streamlit
* PyPDF2
* NLP (TF-IDF, Cosine Similarity)

---

## 📂 Project Structure

```
Smart-Resume-Analyzer/
│
├── app.py
├── streamlit_app.py
├── data/
│   ├── sample_resume.txt
│   └── job_description.txt
├── utils/
│   ├── text_extractor.py
│   ├── text_preprocessor.py
│   ├── tfidf_vectorizer.py
│   ├── similarity.py
│   └── skill_extractor.py
└── README.md
```

---

## ⚙️ How It Works

1. Extracts text from resume (PDF/TXT)
2. Cleans and preprocesses text
3. Computes TF-IDF vectors
4. Calculates cosine similarity → ATS Score
5. Extracts skills and compares with job description
6. Displays matched and missing skills

---

## ▶️ How to Run

```bash
# Install dependencies
pip install streamlit PyPDF2

# Run the app
python -m streamlit run streamlit_app.py
```

---

## 📊 Example Output

* ATS Score: 32.17%
* Missing Skills: backend, system design

---

## 🎯 Future Improvements

* Use advanced NLP models (BERT)
* Improve skill extraction using ML
* Add resume suggestions
* Deploy online

---

## 👨‍💻 Author

Harsh Thakar
