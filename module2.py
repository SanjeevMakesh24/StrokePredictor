import numpy as np
import pandas as pd
import math

    stroke_data = pd.read_csv('healthcare-dataset-stroke-data.csv')

    X = stroke_data.drop(columns=['stroke'])

    ones = [1] * len(stroke_data['stroke'])
    X = X.insert(0, 'x_zero', ones)

    X = np.array(X)

    y = np.array(stroke_data['stroke'])

    X_transpose = X.transpose()

    theta = np.matmul( np.linalg.inv( np.matmul(X_transpose, X) ), (np.matmul(X_transpose, y)) )

    print(theta)
    X_userInput = [] #input by user 
    print(X_userInput)

    h_theta = np.matmul(theta.transpose(), X_userInput)

    LinearRegression = h_theta
    print(LinearRegression)

    e = math.e
    LogisticalRegression = 1 / (1 + pow(e, h_theta))

    print(LogisticalRegression)
