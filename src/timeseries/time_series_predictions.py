import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt


def prepare_data(dataframe, column_to_predict, number_of_predictions):
    x = np.array(dataframe[[column_to_predict]])
    x = preprocessing.scale(x)
    x_lately = x[-number_of_predictions:]
    x = x[:-number_of_predictions]

    label = dataframe[column_to_predict].shift(-number_of_predictions)
    label.dropna(inplace=True)
    y = np.array(label)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, shuffle=False,
                                                                        test_size=number_of_predictions)
    return [x_train, x_test, y_train, y_test, x_lately]


def predict(coin_information_path, column_to_predict, test_percentage):
    dataframe = pd.read_csv(coin_information_path)
    number_of_predictions = int(len(dataframe) * test_percentage)
    x_train, x_test, y_train, y_test, x_lately = prepare_data(dataframe, column_to_predict, number_of_predictions)

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

        accuracy = calculate_average_accuracy(predictions, y_test)
        print("Score of '" + model[0] + "' based on prediction is: " + str(score))
        print("Accuracy is: " + str(accuracy) + "%")

        visualize(dataframe, predictions, number_of_predictions, column_to_predict, model[0])


def calculate_average_accuracy(predicted_data, test_data):
    accuracy_sum = 0
    for i in range(len(test_data)):
        accuracy_sum += (100 * (float(predicted_data[i]) / float(test_data[i])))
    return accuracy_sum / len(test_data)


def visualize(dataframe, predictions, number_of_predictions, column_to_predict, model_name):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    plt.figure()
    plt.plot(dataframe['Date'], dataframe[column_to_predict])
    last = dataframe['Date'].iloc[-number_of_predictions:]
    plt.plot(last, predictions)
    plt.xlabel('                            Date:                                   ' + model_name)
    plt.ylabel(column_to_predict)
    plt.show()
