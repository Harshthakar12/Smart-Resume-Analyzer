import re

# Basic list of stopwords (common useless words)
STOPWORDS = [
    "and", "or", "the", "is", "in", "at", "of", "a", "to", "for", "on"
]

def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenization (split into words)
    words = text.split()

    # 4. Remove stopwords
    filtered_words = [word for word in words if word not in STOPWORDS]

    # 5. Join back to string
    clean_text = " ".join(filtered_words)

    return clean_text
def remove_stopwords(text):
    stopwords = {
        "the", "is", "in", "and", "to", "for", "a", "of",
        "with", "on", "at", "by", "an", "be", "this",
        "that", "are", "as", "it", "from", "or",
        "looking", "software", "engineer", "experience"
    }

    words = text.split()
    filtered = [word for word in words if word not in stopwords]

    return " ".join(filtered)