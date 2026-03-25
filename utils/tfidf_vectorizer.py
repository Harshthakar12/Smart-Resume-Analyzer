import math
from collections import Counter

def compute_tf(text):
    words = text.split()
    word_count = len(words)

    tf_dict = {}

    for word in words:
        tf_dict[word] = tf_dict.get(word, 0) + 1

    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / word_count

    return tf_dict


def compute_idf(documents):
    import math
    N = len(documents)
    idf_dict = {}

    all_words = set(word for doc in documents for word in doc.split())

    for word in all_words:
        count = sum(1 for doc in documents if word in doc.split())
        idf_dict[word] = math.log((1+N)/(1+count))+1

    return idf_dict


def compute_tfidf(tf, idf):
    tfidf = {}

    for word in tf:
        tfidf[word] = tf[word] * idf.get(word, 0)

    return tfidf