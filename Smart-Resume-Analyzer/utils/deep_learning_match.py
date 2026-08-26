from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load AI Model
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(resume_text, jd_text):

    # Convert text into embeddings
    embeddings = model.encode([resume_text, jd_text])

    # Calculate similarity
    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return score