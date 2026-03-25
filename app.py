from utils.text_extractor import extract_text
from utils.text_preprocessor import preprocess_text
from utils.tfidf_vectorizer import compute_tf, compute_idf, compute_tfidf

file_path = "data/sample_resume.txt"

# Step 1: Extract text
text = extract_text(file_path)

# Step 2: Clean text
clean_text = preprocess_text(text)

# Step 3: Prepare documents (for now only 1)
documents = [clean_text]

# Step 4: Compute TF
tf = compute_tf(clean_text)

# Step 5: Compute IDF
idf = compute_idf(documents)

# Step 6: Compute TF-IDF
tfidf = compute_tfidf(tf, idf)

print("----- TF-IDF Scores -----\n")
for word, score in tfidf.items():
    print(f"{word}: {score:.4f}")