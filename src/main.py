from preprocess import generate_crypto_coin_model
from time_series_predictions import predict

if __name__ == '__main__':
    # generate_crypto_coin_model('../data/cryptoInfo/coin_Bitcoin.csv', '../data/mergedData.csv')

    column_to_predict = 'Close'
    predict('../data/cryptoInfo/coin_Bitcoin.csv', column_to_predict, 0.2)
    # predict('../data/cryptoInfo/coin_Bitcoin.csv', column_to_predict_2, 0.1)
