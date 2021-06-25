import nltk
import time

from sklearn.feature_extraction.text import TfidfVectorizer


import numpy as np
import pandas as pd

dataset = pd.read_csv('../data/fullData.csv')
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset['output'] = round(100 * dataset['movement_the_day_after'] / dataset['Close'])
# dataset['message'].apply(lambda x: len(x.split(' '))>1)
# dataset.to_csv('../data/filteredKnnInput.csv')

#dataset['output'] = round(dataset['movement_the_day_after'] / 1000)


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
    def __init__(self, k = 3):
        self.k = k
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


    def predict(self, x_test, start_index):
        self.x_test = x_test
        self.start_index = start_index
        y_predict = []

        for i in range(len(x_test)):
            tic = time.perf_counter()

            max_sim = 0
            max_indexes = []
            max_similarities = []
            pred_result = []
            step = 1000
            j = 0

            try:
                if x_test[i + 3 * train_size + 3] != '' and x_test[i + 3 * train_size + 3] is not None:

                    while (j < self.x_train.shape[0]):
                        if (j + 1 >= self.x_train.shape[0]):
                            j += 1
                        elif (j + step > self.x_train.shape[0]):
                            document = self.x_test[i + 3 * train_size + 3]
                            sim_obj = self.get_most_similar_document(document, j, self.x_train.shape[0] - 1)
                            if sim_obj['temp_sim'] > max_sim:
                                max_similarities.append(sim_obj['temp_sim'])
                                max_indexes.append(sim_obj['temp_index'])
                                max_indexes = sorted(zip(max_similarities, max_indexes), reverse=True)[:self.k]
                            j = self.x_train.shape[0]
                        else:
                            document = self.x_test[i + 3 * train_size + 3]
                            sim_obj = self.get_most_similar_document(document, j, j + step - 1)
                            if sim_obj['temp_sim'] > max_sim:
                                max_similarities.append(sim_obj['temp_sim'])
                                max_indexes.append(sim_obj['temp_index'])
                                pred_result = sorted(zip(max_similarities, max_indexes), reverse=True)[:self.k]

                            j += step

                    temp_sum = 0
                    for index in range(len(pred_result)):
                        temp_sum += self.y_train[pred_result[index][1]]


                    prediction = round(temp_sum / len(pred_result))

                    prediction = get_class(prediction)
                    y_predict.append(prediction)

                    toc = time.perf_counter()
                    #print(f"The operation took {toc - tic:0.4f} seconds")
            except:
                print('opsi')
                print(x_test.head(1))
                print(i + 3 * train_size + 3)
                y_predict.append('neutral')
                continue

        return y_predict

train_size = int((0.995 * len(dataset))/3)
test_size = int(0.005 * len(dataset))
print(f"Train size: {train_size}, Test Size: {test_size}")

train_corpus = dataset[0:train_size - 1]
test_corpus = dataset[-test_size:]

train_corpus2 = dataset[train_size:2*train_size-1]
test_corpus2 = dataset[-test_size:]

train_corpus3 = dataset[2*train_size:3*train_size-1]
test_corpus3 = dataset[-test_size:]

X_train = train_corpus['message']
y_train = train_corpus['output']

X_train2 = train_corpus2['message']
y_train2 = train_corpus2['output']

X_train3 = train_corpus3['message']
y_train3 = train_corpus3['output']

classifier = KNN_NLC_Classifer(k=3)
classifier2 = KNN_NLC_Classifer(k=3)
classifier3 = KNN_NLC_Classifer(k=3)

classifier.fit(X_train, y_train)
classifier2.fit(X_train2, y_train2)
classifier3.fit(X_train3, y_train3)

y_pred_final = classifier.predict(test_corpus['message'], 0)
print('First finished')
y_pred_final2 = classifier.predict(test_corpus['message'], train_size)
print('Second finished')
y_pred_final3 = classifier.predict(test_corpus['message'], 2*train_size)
print('Third finished')



num_all = len(test_corpus)
num_correct = 0

for j in range(len(y_pred_final)):
    prediction_list = [y_pred_final[j], y_pred_final2[j], y_pred_final3[j]]
    result = max(set(prediction_list), key=prediction_list.count)
    print(result)
    actual_output = 'neutral'
    try:
        actual_output = get_class(test_corpus.loc[3*train_size + j + 3, 'output'])
    except:
        actual_output = 'neutral'
        continue
    print(get_class(test_corpus.loc[3*train_size + j + 3, 'output']))
    if result == get_class(test_corpus.loc[3*train_size + j + 3, 'output']):
    #if y_pred_final[j] == test_corpus.loc[train_size + j + 1, 'output']:

        num_correct += 1

print(num_correct/num_all)

# for j in range(2):
#     prediction_list = [1, 2, 3]
#     result = max(set(prediction_list), key=prediction_list.count)
#     print(result)
# #     if result == get_class(test_corpus.loc[3*train_size + j + 3, 'output']):
# #     #if y_pred_final[j] == test_corpus.loc[train_size + j + 1, 'output']:
# #
# #         num_correct += 1
# #
# # print(num_correct/num_all)