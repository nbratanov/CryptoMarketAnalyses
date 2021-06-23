import nltk
import numpy as np
import pandas as pd
from nltk import PorterStemmer

from src.common.utils import Utils

stemmer = PorterStemmer()


def build_training_data(data_path):
    print("Building classes out of data from " + data_path)
    data = pd.read_csv(data_path)
    normalized_data = Utils.normalize_data(data)

    message = np.array(normalized_data['message'])
    price_increase = []
    for i in range(len(normalized_data)):
        increase = 100 * float(normalized_data.loc[i, 'movement_the_day_after']) / float(normalized_data.loc[i, 'Open'])
        price_increase.append(increase)
    print("Collected evaluating the percentage of price increase in a day")

    training_data = []
    for i in range(len(normalized_data)):
        if price_increase[i] <= -5:
            training_data.append({"class": "huge_loss", "message": message[i]})
        if -5 < price_increase[i] <= -1:
            training_data.append({"class": "loss", "message": message[i]})
        if -1 < price_increase[i] <= 1:
            training_data.append({"class": "neutral", "message": message[i]})
        if 1 < price_increase[i] <= 5:
            training_data.append({"class": "profit", "message": message[i]})
        if price_increase[i] > 5:
            training_data.append({"class": "huge_profit", "message": message[i]})
    print("Training data was created")

    return training_data


def classify_unique_words(training_data):
    corpus_words = {}
    class_words = {}

    classes = ["huge_loss", "loss", "neutral", "profit", "huge_profit"]
    for c in classes:
        class_words[c] = []

    for data in training_data:
        for word in nltk.word_tokenize(data['message']):
            if word not in ["?", "'s", "."]:
                if word not in corpus_words:
                    corpus_words[word] = 1
                else:
                    corpus_words[word] += 1

                class_words[data['class']].extend([word])

    # print("Corpus words and counts: " + corpus_words)
    # print("Class words: " + class_words)
    return class_words, corpus_words


# Calculates the score of a message based on number of occurrences of a word in a class
def calculate_class_score(message, class_words, class_name):
    score = 0
    for word in nltk.word_tokenize(message):
        stemmed_word = stemmer.stem(word.lower())
        if stemmed_word in class_words[class_name]:
            score += 1
    return score


def find_highest_class_score(message, class_words):
    for c in class_words.keys():
        score = calculate_class_score(message, class_words, c)
        print("Class: " + c + " Score: " + str(score) + "\n")


# Calculates the score of a message based both number of occurrences of a word in a class and the word commonality
def calculate_class_score_commonality(message, class_words, corpus_words, class_name, show_details=False):
    score = 0
    for word in nltk.word_tokenize(message):
        stemmed_word = stemmer.stem(word.lower())
        if stemmed_word in class_words[class_name]:
            score += (1 / corpus_words[stemmed_word])
            if show_details:
                print("   match: " + stemmed_word + "(" + str(1 / corpus_words[stemmed_word]) + ")")
    return score


def find_highest_class_score_commonality(sentence, class_words, corpus_words):
    for c in class_words.keys():
        score = calculate_class_score_commonality(sentence, class_words, corpus_words, c)
        print("Class: " + c + " Score: " + str(score) + "\n")


def classify(sentence, class_words, corpus_words, show_details=False):
    high_class = None
    high_score = 0
    for c in class_words.keys():
        score = calculate_class_score_commonality(sentence, class_words, corpus_words, c, show_details)
        if score > high_score:
            high_class = c
            high_score = score

    return high_class, high_score


def test_classification(data_path):
    data = build_training_data(data_path)
    class_words, corpus_words = classify_unique_words(data)

    msg = "Bitcoin just went up. Invest"
    # calculate_class_score(msg, class_words, "neutral")
    # find_highest_class_score(msg, class_words)
    # calculate_class_score_commonality(msg, class_words, corpus_words)
    # find_highest_class_score_commonality(msg, class_words, corpus_words)

    print(classify(msg, class_words, corpus_words))
    print(classify("invest now in bitcoin", class_words, corpus_words))
    print(classify("BTC is up", class_words, corpus_words))
    print(classify("btc is awful, right now", class_words, corpus_words))
    print(classify("you should sell your bitcoin", class_words, corpus_words))
    print(classify("bitcoin is about to increase", class_words, corpus_words))
    print(classify("bitcoin is about to rise", class_words, corpus_words))
    print(classify("bitcoin will rise", class_words, corpus_words))
    print(classify("selling bitcoin", class_words, corpus_words))
