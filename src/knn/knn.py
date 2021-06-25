import nltk
import time

from sklearn.feature_extraction.text import TfidfVectorizer


import numpy as np
import pandas as pd

# dataset = pd.read_csv('../../data/fullData.csv')
#dataset = dataset.iloc[::-1].reset_index()
#dataset = dataset.sample(frac=1).reset_index(drop=True)
#dataset['message'].apply(lambda x: len(x.split(' '))>1)
# dataset['output'] = round(100 * dataset['movement_the_day_after'] / dataset['Close'])




def get_class(percent):
    result = 0
    if percent >= -1 and percent <= 1:
        result = 'neutral'
    elif percent > 1 and percent <= 5:
        result = 'profit'
    elif percent > 5:
        result = 'huge_profit'
    elif percent < -1 and percent >= -5:
        result = 'loss'
    elif percent < -5:
        result = 'huge_loss'

    return result


class KNN_NLC_Classifer():
    def __init__(self, k, train_size):
        self.k = k
        self.train_size = train_size
        self.vectorizer = TfidfVectorizer(stop_words='english')


    # This function is used for training
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_most_similar_document(self, document, start_index, end_index):
        sample = dataset['message'][start_index:end_index]
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

    # def get_alt_similarity(document, start_index, end_index):
    #     nlp(u'Hello hi there!')


    def predict(self, x_test):
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
                            max_indexes = sorted(zip(max_similarities, max_indexes), reverse=True)[:self.k]
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
            for index in range(len(pred_result)):
                #pred_result.loc[index] = get_class(round(y_train[pred_result[index][1]]))
                # print(y_train[pred_result[index][1]])
                temp_sum += self.y_train[pred_result[index][1]]

            #prediction = max(set(pred_result), key=pred_result.count)

            value = round(temp_sum/len(pred_result))


            prediction = get_class(value)
            y_predict.append(prediction)
            print(prediction)
            print(value)
            #print(get_class(self.x_test.loc[self.train_size + i + 1, 'output']))

            #print(prediction == get_class(self.x_test.loc[self.train_size + i + 1, 'output']))

            toc = time.perf_counter()
            print(f"The operation took {toc - tic:0.4f} seconds")
        return y_predict


def check_knn_accuracy():
    train_size = int(0.998 * len(dataset))
    test_size = int(0.002 * len(dataset))
    print(f"Train size: {train_size}, Test Size: {test_size}")

    train_corpus = dataset[:train_size]
    test_corpus = dataset[-test_size:]

    X_train = train_corpus['message']
    y_train = train_corpus['output']


    classifier = KNN_NLC_Classifer(3, train_size)

    classifier.fit(X_train, y_train)

    print(test_corpus.head())

    y_pred_final = classifier.predict(test_corpus['message'])

    num_all = len(test_corpus)
    num_correct = 0

    for j in range(len(y_pred_final)):
        if y_pred_final[j] == get_class(test_corpus.loc[train_size + j + 1, 'output']):
            num_correct += 1

    print(num_correct/num_all)

def demo_knn(data_path):
    train_corpus = pd.read_csv(data_path)
    X_train = train_corpus['message']
    y_train = train_corpus['output']

    test_corpus = [
        'Bitcoin to the moon',
        'I love Bitcoin',
        'Bitcoin will go up',
        'Bitcoin will collapse'
    ]

    classifier = KNN_NLC_Classifer(1, -1)
    classifier2 = KNN_NLC_Classifer(5, -1)

    classifier.fit(X_train, y_train)
    classifier.predict(test_corpus)

    classifier2.fit(X_train, y_train)
    classifier2.predict(test_corpus)
