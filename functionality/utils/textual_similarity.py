from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize global TF-IDF Vectorizer to avoid redundant instantiation
vectorizer = TfidfVectorizer()

def text_similarity(text1, text2):
    """
    Compute cosine similarity between two texts using TF-IDF.

    Args:
        text1 (str): First text.
        text2 (str): Second text.

    Returns:
        float: Cosine similarity between the two texts.
    """
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]