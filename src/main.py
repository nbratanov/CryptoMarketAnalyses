import datetime
import pandas as pd

from src.knn.knn import demo_knn
from src.timeseries.time_series_predictions import predict
from src.classification.multinomial_naive_bayes import test_classification_predictions, test_classification

if __name__ == '__main__':
    # generate_crypto_coin_model('../data/cryptoInfo/coin_Bitcoin.csv', '../data/mergedData.csv', Utils.BITCOIN_NAMES)
    # generate_crypto_coin_model('../data/cryptoInfo/coin_Ethereum.csv', '../data/ethereum-data.csv', Utils.ETHEREUM_NAMES)

    now = datetime.datetime.now()
    column_to_predict = 'Close'
    predict('../data/cryptoInfo/coin_Bitcoin.csv', column_to_predict, 0.1)
    print("Estimated time: " + str(datetime.datetime.now() - now))
    # predict('../data/cryptoInfo/coin_Bitcoin.csv', column_to_predict_2, 0.1)

    now = datetime.datetime.now()
    test_classification('../data/clean_data.csv')
    print("Estimated time: " + str(datetime.datetime.now() - now))

    # demo_knn('../data/data-correct.csv')

    # test_classification_predictions('../data/data-correct.csv')
    # test_classification_predictions('../data/bitcoin-data.csv')
    # test_classification_predictions('../data/ethereum-data.csv')


    #### fix data
    # data = pd.read_csv('../data/mergedData-boyan.csv')
    # movement = data['movement_the_day_after']
    # for i in range(len(movement)):
    #     data.loc[i, 'movement_the_day_after'] *= -1
    # data.to_csv('../data/data-correct.csv')

