import pandas as pd

def load_data():
    patients = pd.read_csv('/Users/thomas/Desktop/data/healthcare_recommender/datasets/patients.csv')
    doctors = pd.read_csv('/Users/thomas/Desktop/data/healthcare_recommender/datasets/doctors.csv')
    interactions = pd.read_csv('/Users/thomas/Desktop/data/healthcare_recommender/datasets/interactions.csv')
    return patients, doctors, interactions
