from model.preprocess_data import load_data
from model.collaborative_filtering import train_cf_model, get_top_n_recommendations_cf
from model.content_based_filtering import train_cbf_model, get_top_n_recommendations_cbf
from model.hybrid_recommender import get_hybrid_recommendations


def main():
    patients, doctors, interactions = load_data()
    cf_model, cf_rmse = train_cf_model(interactions)
    tfidf_vectorizer, cosine_similarities = train_cbf_model(doctors)

    patient_id = 1
    print(
        f"CF Top 5 Recommendations for Patient {patient_id}: {get_top_n_recommendations_cf(cf_model, patient_id, interactions)}"
    )
    print(
        f"CBF Top 5 Recommendations for Patient {patient_id}: {get_top_n_recommendations_cbf(patient_id, interactions, doctors, cosine_similarities)}"
    )
    print(
        f"Hybrid Top 5 Recommendations for Patient {patient_id}: {get_hybrid_recommendations(patient_id, cf_model, interactions, doctors, cosine_similarities)}"
    )


if __name__ == "__main__":
    main()
