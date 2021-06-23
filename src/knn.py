import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
nltk.download('genesis')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
genesis_ic = wn.ic(genesis, False, 0.0)
import time

from sklearn.feature_extraction.text import TfidfVectorizer


import numpy as np
import pandas as pd

dataset = pd.read_csv('../data/knnInput.csv')
print(np.max(dataset['movement_the_day_after']))
print(np.mean(dataset['movement_the_day_after']))
print(np.min(dataset['movement_the_day_after']))
dataset['output'] = round(dataset['movement_the_day_after'] / 1000)

class KNN_NLC_Classifer():
    def __init__(self, k = 1, distance_type = 'path'):
        self.k = k
        self.distance_type = distance_type
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
        # no need to normalize, sin
        # ce Vectorizer will return normalized tf-idf
        result = dict()
        result['temp_sim'] = arr[end_index - start_index][result_idx]
        result['temp_index'] = start_index + result_idx
        return result

    # This function runs the K(1) nearest neighbour algorithm and
    # returns the label with closest match.
    def predict(self, x_test):
        self.x_test = x_test
        y_predict = []

        for i in range(len(x_test)):
            tic = time.perf_counter()

            max_sim = 0
            max_indexes = []
            max_similarities = []
            pred_result = []
            step = 7000
            j = 0

            if x_test[i + train_size + 1] != '' and x_test[i + train_size + 1] is not None:
                tic = time.perf_counter()

                while(j < self.x_train.shape[0]):
                    if (j +1 >= self.x_train.shape[0]):
                        j += 1
                    elif (j + step > self.x_train.shape[0]):
                        sim_obj = self.get_most_similar_document(x_test[i + train_size + 1], j, self.x_train.shape[0] - 1)
                        if sim_obj['temp_sim'] > max_sim:
                            max_similarities.append(sim_obj['temp_sim'])
                            max_indexes.append(sim_obj['temp_index'])
                            max_indexes = sorted(zip(max_similarities, max_indexes), reverse=True)[:self.k]
                        j = self.x_train.shape[0]
                    else:
                        sim_obj = self.get_most_similar_document(x_test[i + train_size + 1], j, j + step - 1)
                        if sim_obj['temp_sim'] > max_sim:
                            max_similarities.append(sim_obj['temp_sim'])
                            max_indexes.append(sim_obj['temp_index'])
                            pred_result = sorted(zip(max_similarities, max_indexes), reverse=True)[:self.k]

                        j += step

            toc = time.perf_counter()
            temp_sum = 0
            for index in range(len(pred_result)):
                temp_sum += self.y_train[pred_result[index][1]]

            prediction = round(temp_sum/len(pred_result))
            print(f"Pairwise built in {toc - tic:0.4f} seconds")
            y_predict.append(prediction)

            print(prediction)
            print(test_corpus.loc[train_size + i + 1, 'output'])
            print(prediction == test_corpus.loc[train_size + i + 1, 'output'])
            toc = time.perf_counter()
            print(f"The operation took {toc - tic:0.4f} seconds")
        return y_predict

train_size = int(0.999 * len(dataset))
test_size = int(0.001 * len(dataset))
print(test_size)

train_corpus = dataset[:train_size]
test_corpus = dataset[-test_size:]
X_train = train_corpus['message']

y_train = train_corpus['output']


classifier = KNN_NLC_Classifer(k=3, distance_type='path')
classifier.fit(X_train, y_train)

y_pred_final = classifier.predict(test_corpus['message'])

num_all = len(test_corpus)
num_correct = 0

for j in range(len(y_pred_final)):
    if y_pred_final[j] == test_corpus.loc[train_size + j + 1, 'output']:
        num_correct += 1

print(num_correct/num_all)