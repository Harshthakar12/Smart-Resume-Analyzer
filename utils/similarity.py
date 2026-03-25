import math

def cosine_similarity(vec1, vec2):
    dot_product = 0
    norm1 = 0
    norm2 = 0

    all_words = set(vec1.keys()).union(set(vec2.keys()))

    for word in all_words:
        v1 = vec1.get(word, 0)
        v2 = vec2.get(word, 0)

        dot_product += v1 * v2
        norm1 += v1 ** 2
        norm2 += v2 ** 2

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))