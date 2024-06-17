from .collaborative_filtering import get_top_n_recommendations_cf
from .content_based_filtering import get_top_n_recommendations_cbf


def get_hybrid_recommendations(
    patient_id, cf_model, interactions, doctors, cosine_similarities, n=5, alpha=0.5
):
    cf_recommendations = get_top_n_recommendations_cf(
        cf_model, patient_id, interactions, n
    )
    cbf_recommendations = get_top_n_recommendations_cbf(
        patient_id, interactions, doctors, cosine_similarities, n
    )
    combined_recommendations = {}

    for doctor, score in cf_recommendations:
        combined_recommendations[doctor] = alpha * score

    for doctor, score in cbf_recommendations:
        if doctor in combined_recommendations:
            combined_recommendations[doctor] += (1 - alpha) * score
        else:
            combined_recommendations[doctor] = (1 - alpha) * score
    top_n_combined = sorted(
        combined_recommendations.items(), key=lambda x: x[1], reverse=True
    )[:n]
    return top_n_combined
