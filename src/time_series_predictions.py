import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt


def prepare_data(dataframe, column_to_predict, number_of_predictions, test_size):
    x = np.array(dataframe[[column_to_predict]])
    x = preprocessing.scale(x)
    x_lately = x[-number_of_predictions:]
    x = x[:-number_of_predictions]

    label = dataframe[column_to_predict].shift(-number_of_predictions)
    label.dropna(inplace=True)
    y = np.array(label)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size)
    return [x_train, x_test, y_train, y_test, x_lately]


def predict(coin_information_path, column_to_predict, test_percentage):
    dataframe = pd.read_csv(coin_information_path)
    number_of_predictions = int(len(dataframe) * test_percentage)
    x_train, x_test, y_train, y_test, x_lately = prepare_data(dataframe, column_to_predict, number_of_predictions,
                                                              test_percentage)

    # Try out different models suitable for time series forecasting
    models = [
        ('Linear Regression', LinearRegression()),
        ('K-Neighbours', KNeighborsRegressor()),
        ('Neural Network', MLPRegressor()),
        ('Random Forest', RandomForestRegressor())
    ]

    for model in models:
        learner = model[1]
        learner.fit(x_train, y_train)
        score = learner.score(x_test, y_test)
        predictions = learner.predict(x_lately)
        print("Score of '" + model[0] + "' based on prediction is: " + str(score))

        visualize(dataframe, predictions, number_of_predictions, column_to_predict)


def visualize(dataframe, forecast, number_of_forecasts, column_to_predict):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    plt.figure()
    plt.plot(dataframe['Date'], dataframe[column_to_predict])
    last = dataframe['Date'].iloc[-number_of_forecasts:]
    plt.plot(last, forecast)
    plt.xlabel('Date')
    plt.ylabel(column_to_predict)
    plt.show()
