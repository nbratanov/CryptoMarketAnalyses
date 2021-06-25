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
from nltk.tokenize import word_tokenize

dataset = pd.read_csv('../data/knnInput.csv')
print(np.max(dataset['movement_the_day_after']))
print(np.mean(dataset['movement_the_day_after']))
print(np.min(dataset['movement_the_day_after']))
dataset['output'] = round(dataset['movement_the_day_after'] / 1000)
#dataset = dataset.sample(frac=1).reset_index(drop=True)

def get_class(percent):
    result = 0
    if percent >= -1 and percent <= 1:
        result = 'neutral'
    elif percent > 1 and percent <= 4:
        result = 'profit'
    elif percent > 4:
        result = 'huge_profit'
    elif percent < -1 and percent >= -4:
        result = 'loss'
    elif percent < -4:
        result = 'huge_loss'

    return result


class KNN_NLC_Classifer():
    def __init__(self, k = 1, distance_type = 'path'):
        self.k = k
        self.distance_type = distance_type
        self.vectorizer = TfidfVectorizer(stop_words='english')


    # This function is used for training
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def document_similarity(self, doc1, doc2):
        synsets1 = self.doc_to_synsets(doc1)
        synsets2 = self.doc_to_synsets(doc2)

        return (self.similarity_score(synsets1, synsets2) + self.similarity_score(synsets2, synsets1)) / 2

    # This function runs the K(1) nearest neighbour algorithm and
    # returns the label with closest match.
    def predict(self, x_test):
        self.x_test = x_test
        y_predict = []
        count_correct = 0
        for i in range(len(x_test)):
            tic = time.perf_counter()

            max_sim = 0
            max_index = 0
            for j in range(self.x_train.shape[0]):
                temp = self.document_similarity(x_test[i + train_size +1], self.x_train[j])
                if temp > max_sim:
                    max_sim = temp
                    max_index = j
            y_predict.append(self.y_train[max_index])
            print(self.y_train[max_index] == test_corpus.loc[i + train_size+1, 'output'])
            if self.y_train[max_index] == test_corpus.loc[i + train_size+1, 'output']:
                count_correct += 1
            toc = time.perf_counter()
            print(f"This is how it takes {toc - tic:0.4f} seconds")
            print(f'{count_correct}/{i}')

        return y_predict

    def convert_tag(self, tag):
        tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
        try:
            return tag_dict[tag[0]]
        except KeyError:
            return None

    def doc_to_synsets(self, doc):
        tokens = word_tokenize(doc + ' ')

        l = []
        tags = nltk.pos_tag([tokens[0] + ' ']) if len(tokens) == 1 else nltk.pos_tag(tokens)

        for token, tag in zip(tokens, tags):
            syntag = self.convert_tag(tag[1])
            syns = wn.synsets(token, syntag)
            if (len(syns) > 0):
                l.append(syns[0])
        return l

    def similarity_score(self, s1, s2, distance_type='path'):
        s1_largest_scores = []

        for i, s1_synset in enumerate(s1, 0):
            max_score = 0
            for s2_synset in s2:
                if distance_type == 'path':
                    score = s1_synset.path_similarity(s2_synset, simulate_root=False)
                else:
                    score = s1_synset.wup_similarity(s2_synset)
                if score != None:
                    if score > max_score:
                        max_score = score

            if max_score != 0:
                s1_largest_scores.append(max_score)

        mean_score = np.mean(s1_largest_scores)

        return mean_score

train_size = int(0.999 * len(dataset))
test_size = int(0.001 * len(dataset))
print(test_size)

train_corpus = dataset[:train_size]
test_corpus = dataset[-test_size:]
X_train = train_corpus['message']

y_train = train_corpus['output']


classifier = KNN_NLC_Classifer(k=1, distance_type='path')
classifier.fit(X_train, y_train)

y_pred_final = classifier.predict(test_corpus['message'])

num_all = len(test_corpus)
num_correct = 0

for j in range(len(y_pred_final)):
    if y_pred_final[j] == test_corpus.loc[train_size + j + 1, 'output']:
        num_correct += 1

print(num_correct/num_all)