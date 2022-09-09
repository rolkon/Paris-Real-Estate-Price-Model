import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skmetrics
import json


def train_model(location_data):
    
    

    # prepare data for training and testing, assume 80% training data and 20% testing data
    X1 = location_data['PosX'].to_numpy(dtype=float)
    X2 = location_data['PosY'].to_numpy(dtype=float)
    X = np.hstack([X1[..., np.newaxis], X2[..., np.newaxis]])

    train_fraction = 0.8
    train_test_index = int(X.shape[0] * 0.8)

    y = location_data['CostPerArea'].to_numpy(dtype=float)

    X_train = X[:train_test_index]
    X_test = X[train_test_index:]

    y_train = y[:train_test_index]
    y_test = y[train_test_index:]

    # use KNN Regressor and GridSearch with cross validation
    # use median absolute error as metric to avoid large outliers and get a representative value for the price error
    model = KNeighborsRegressor(weights='distance')

    gs_model = GridSearchCV(model,
                            param_grid={'n_neighbors':range(1,50)},
                            n_jobs=1,
                            scoring='neg_median_absolute_error')

    gs_model.fit(X_train, y_train)

    test_error = skmetrics.median_absolute_error(y_test, gs_model.predict(X_test))

    return gs_model, test_error

def main():
    
    location_data = pd.read_csv('../data/cost_location_data.csv')

    print("Training model...")
    gs_model, test_error = train_model(location_data)
    
    # save model data
    with open('../data/model_params.json', 'w') as outfile:
        json.dump(gs_model.best_params_, outfile)

    print('Fitting done. Best value for k:', gs_model.best_params_)
    print('Median absolute error on test set:', test_error)

if __name__ == '__main__':
    main()