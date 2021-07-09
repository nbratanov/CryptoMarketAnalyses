import nltk
import time

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from src.common.utils import Utils

dataset = pd.read_csv('../../data/fullData.csv')
#dataset = dataset[dataset['date'].apply(lambda x:  pd.to_datetime(x) > pd.to_datetime('2019-01-01 00:00:00+00:00'))]
#dataset = dataset.sample(frac=1).reset_index(drop=True)
#dataset['message'].apply(lambda x: len(x.split(' '))>1)
# dataset['output'] = round(100 * dataset['movement_the_day_after'] / dataset['Close'])

class KNN_NLC_Classifer():
    def __init__(self, k, train_size, dataset):
        self.k = k
        self.train_size = train_size
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.dataset = dataset


    # This function is used for training
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_most_similar_document(self, document, start_index, end_index):
        sample = self.dataset['message'][start_index:end_index]
        sample.loc[end_index + 1] = document
        tfidf = self.vectorizer.fit_transform(sample.values)
        pairwise_similarity = tfidf * tfidf.T
        arr = pairwise_similarity.toarray()
        np.fill_diagonal(arr, np.nan)
        result_idx = np.nanargmax(arr[end_index - start_index])

        result = dict()
        result['temp_sim'] = arr[end_index - start_index][result_idx]

        result['temp_index'] = start_index + result_idx
        return result

    def predict(self, x_test, use_mean):
        self.x_test = x_test
        y_predict = []

        for i in range(len(x_test)):
            tic = time.perf_counter()

            max_sim = 0
            max_indexes = []
            max_similarities = []
            pred_result = []
            step = 2000
            j = 0

            if x_test[i + self.train_size + 1] != '' and x_test[i + self.train_size + 1] is not None:

                while(j < self.x_train.shape[0]):
                    if (j +1 >= self.x_train.shape[0]):
                        j += 1
                    elif (j + step > self.x_train.shape[0]):
                        document = self.x_test[i + self.train_size + 1]
                        sim_obj = self.get_most_similar_document(document, j, self.x_train.shape[0] - 1)
                        if sim_obj['temp_sim'] > max_sim:
                            max_similarities.append(sim_obj['temp_sim'])
                            max_indexes.append(sim_obj['temp_index'])
                            pred_result = sorted(zip(max_similarities, max_indexes), reverse=True)[:self.k]
                        j = self.x_train.shape[0]
                    else:
                        document = self.x_test[i + self.train_size + 1]
                        sim_obj = self.get_most_similar_document(document, j, j + step - 1)
                        if sim_obj['temp_sim'] > max_sim:
                            max_similarities.append(sim_obj['temp_sim'])
                            max_indexes.append(sim_obj['temp_index'])
                            pred_result = sorted(zip(max_similarities, max_indexes), reverse=True)[:self.k]

                        j += step

            temp_sum = 0
            pred_array = []
            for index in range(len(pred_result)):
                #pred_result.loc[index] = Utils.get_class(round(self.y_train[pred_result[index][1]]))
                print(self.y_train[pred_result[index][1]])
                temp_sum += self.y_train[pred_result[index][1]]
                pred_array.append(Utils.get_class(self.y_train[pred_result[index][1]]))
            #prediction = max(set(pred_result), key=pred_result.count)

            if use_mean:
                value = temp_sum/len(pred_result)
                print(value)
                prediction = Utils.get_class(value)
            else:
                prediction = max(set(pred_array), key=pred_array.count)

            y_predict.append(prediction)

            toc = time.perf_counter()
            print(f"The operation took {toc - tic:0.4f} seconds")
        return y_predict


def check_knn_accuracy(data_path):
    dataset = pd.read_csv(data_path)
    dataset['output'] = round(100 * dataset['movement_the_day_after'] / dataset['Close'])
    train_size = int(0.998 * len(dataset))
    test_size = int(0.002 * len(dataset))
    print(f"Train size: {train_size}, Test Size: {test_size}")

    train_corpus = dataset[:train_size]
    test_corpus = dataset[-test_size:]

    X_train = train_corpus['message']
    y_train = train_corpus['output']


    classifier = KNN_NLC_Classifer(3, train_size, dataset)

    classifier.fit(X_train, y_train)

    print(test_corpus.head())

    y_pred_final = classifier.predict(test_corpus['message'], False)

    num_all = len(test_corpus)
    num_correct = 0

    for j in range(len(y_pred_final)):
        if y_pred_final[j] == Utils.get_class(test_corpus.loc[train_size + j + 1, 'output']):
            num_correct += 1

    print(num_correct/num_all)

def demo_knn(data_path):
    train_corpus = pd.read_csv(data_path)
    train_corpus['output'] = 100 * train_corpus['movement_the_day_after'] / train_corpus['Close']
    X_train = train_corpus['message']
    y_train = train_corpus['output']

    test_corpus = [
        'I love Bitcoin',
        'Bitcoin will go up',
        'Bitcoin will collapse',
        'Sell Bitcoin now'
    ]

    classifier = KNN_NLC_Classifer(1, -1, train_corpus)
    classifier2 = KNN_NLC_Classifer(5, -1, train_corpus)

    print("Print for k=1 nearest neighbours")
    classifier.fit(X_train, y_train)
    classifier.predict(test_corpus, False)
    print("\n")

    print("Print for k=5 nearest neighbours")
    classifier2.fit(X_train, y_train)
    classifier2.predict(test_corpus, False)
    print("\n")

check_knn_accuracy('../../data/data/bitcoin-data.csv')
