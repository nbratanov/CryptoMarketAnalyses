from preprocess import generate_crypto_coin_model
from time_series_predictions import predict
from classification.multinomial_naive_bayes import test_classification

if __name__ == '__main__':
    # generate_crypto_coin_model('../data/cryptoInfo/coin_Bitcoin.csv', '../data/mergedData.csv')

    # column_to_predict = 'Close'
    # predict('../data/cryptoInfo/coin_Bitcoin.csv', column_to_predict, 0.2)
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

    test_classification('../data/mergedData.csv')


