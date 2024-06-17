from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


def train_cf_model(interactions):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        interactions[["patient_id", "doctor_id", "rating"]], reader
    )
    trainset, testset = train_test_split(data, test_size=0.25)
    cf_model = SVD()
    cf_model.fit(trainset)
    predictions = cf_model.test(testset)
    rmse = accuracy.rmse(predictions)
    return cf_model, rmse


def get_top_n_recommendations_cf(model, patient_id, interactions, n=5):
    all_doctors = interactions["doctor_id"].unique()
    rated_doctors = interactions[interactions["patient_id"] == patient_id][
        "doctor_id"
    ].unique()
    unrated_doctors = [doctor for doctor in all_doctors if doctor not in rated_doctors]
    predicted_ratings = [
        model.predict(patient_id, doctor).est for doctor in unrated_doctors
    ]
    top_n_doctors = sorted(
        zip(unrated_doctors, predicted_ratings), key=lambda x: x[1], reverse=True
    )[:n]
    return top_n_doctors
