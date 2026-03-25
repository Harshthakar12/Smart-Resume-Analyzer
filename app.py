from utils.text_extractor import extract_text
from utils.tfidf_vectorizer import compute_tf, compute_idf, compute_tfidf
from utils.similarity import cosine_similarity
from utils.text_preprocessor import preprocess_text, remove_stopwords
# Files
resume_path = "data/sample_resume.txt"
jd_path = "data/job_description.txt"

# Extract
resume_text = extract_text(resume_path)
jd_text = extract_text(jd_path)

# Clean
resume_clean = remove_stopwords(preprocess_text(resume_text))
jd_clean = remove_stopwords(preprocess_text(jd_text))

# Documents for IDF
documents = [resume_clean, jd_clean]

# TF
tf_resume = compute_tf(resume_clean)
tf_jd = compute_tf(jd_clean)

# IDF
idf = compute_idf(documents)

# TF-IDF
tfidf_resume = compute_tfidf(tf_resume, idf)
tfidf_jd = compute_tfidf(tf_jd, idf)

# Similarity
score = cosine_similarity(tfidf_resume, tfidf_jd)

print(f"\nATS Score: {round(score * 100, 2)}%")