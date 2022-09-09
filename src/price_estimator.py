import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import json

def main():
    print('Preparing...')
    
    with open('../data/model_params.json', 'r') as infile:
        model_params = json.load(infile)
    
    data = pd.read_csv('../data/cost_location_data.csv')
    
    X1 = data['PosX'].to_numpy(dtype=float)
    X2 = data['PosY'].to_numpy(dtype=float)
    y = data['CostPerArea'].to_numpy(dtype=float)

    X = np.hstack([X1[..., np.newaxis], X2[..., np.newaxis]])

    model = KNeighborsRegressor(n_neighbors=model_params['n_neighbors'])
    model.fit(X, y)
    
    x_min = np.round(np.min(X[:,0]), 5)
    x_max = np.round(np.max(X[:,0]), 5)
    y_min = np.round(np.min(X[:,1]), 5)
    y_max = np.round(np.max(X[:,1]), 5)
    
    x_coord = input('Enter x-coordinate (accepted range [{}-{}]):\n'.format(x_min, x_max))
    y_coord = input('Enter y-coordinate (accepted range [{}-{}]):\n'.format(y_min, y_max))
    
    x_coord = float(x_coord)
    y_coord = float(y_coord)
    
    if x_min <= x_coord <= x_max and y_min <= y_coord <= y_max:

        prediction = model.predict([[x_coord, y_coord]])
        print('Predicted price per area for chosen coordinates:', prediction)
        
    else:
        print("Entered coordinates not between range.")

if __name__ == '__main__':
    main()