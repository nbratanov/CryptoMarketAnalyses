from src.timeseries.time_series_predictions import predict

if __name__ == '__main__':
    # generate_crypto_coin_model('../data/cryptoInfo/coin_Bitcoin.csv', '../data/mergedData.csv', Utils.BITCOIN_NAMES)
    # generate_crypto_coin_model('../data/cryptoInfo/coin_Ethereum.csv', '../data/ethereum-data.csv', Utils.ETHEREUM_NAMES)
    # generate_crypto_coin_model('../data/cryptoInfo/coin_Bitcoin.csv', '../data/bitcoin-data.csv', Utils.ETHEREUM_NAMES)

    column_to_predict = 'Close'
    predict('../data/cryptoInfo/coin_Bitcoin.csv', column_to_predict, 0.1)
    # predict('../data/cryptoInfo/coin_Ethereum.csv', column_to_predict, 0.1)
    # predict('../data/cryptoInfo/coin_Bitcoin.csv', column_to_predict_2, 0.1)

    # Testing most price profit/loss
    # dataset = pd.read_csv('../../data/fullData.csv')
    # idx_max = dataset['movement_the_day_after'].argmax()
    # idx_min = dataset['movement_the_day_after'].argmin()
    # row = dataset.loc[idx_max]
    # row_min = dataset.loc[idx_min]
    # print(str(100 * float(row['movement_the_day_after'])/float(row['Open'])) + "%")
    # print(str(100 * float(row_min['movement_the_day_after'])/float(row_min['Open'])) + "%")
    # print(np.mean(dataset['movement_the_day_after']))
    # print(np.min(dataset['movement_the_day_after']))

    # test_classification('../data/mergedData.csv')
    # test_classification_predictions('../data/ethereum-data.csv')


