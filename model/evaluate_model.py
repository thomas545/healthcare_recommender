from surprise import cross_validate

def evaluate_cf_model(cf_model, data):
    results = cross_validate(cf_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return results
