from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def train_cbf_model(doctors):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(doctors["specialty"])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    return tfidf_vectorizer, cosine_similarities


def get_top_n_recommendations_cbf(
    patient_id, interactions, doctors, cosine_similarities, n=5
):
    patient_interactions = interactions[interactions["patient_id"] == patient_id]
    doctor_indices = patient_interactions["doctor_id"].values
    similarity_scores = cosine_similarities[doctor_indices].mean(axis=0)
    top_n_indices = similarity_scores.argsort()[-n:][::-1]
    return [(doctors.iloc[i]["doctor_id"], similarity_scores[i]) for i in top_n_indices]
