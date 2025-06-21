from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords(corpus, top_k=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    feature_names = vectorizer.get_feature_names_out()

    all_keywords = []

    for row in tfidf_matrix:
        row_array = row.toarray().flatten()
        top_indices = row_array.argsort()[-top_k:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        all_keywords.append(keywords)

    return all_keywords
