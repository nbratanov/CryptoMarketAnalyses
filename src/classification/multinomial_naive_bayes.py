from datetime import datetime

import nltk
import numpy as np
import pandas as pd

from src.common.utils import Utils


def build_classified_data(data):
    print("Building classes out of data")
    normalized_data = Utils.normalize_data(data)

    message = np.array(normalized_data['message'])
    price_increase = []
    for i in range(len(normalized_data)):
        increase = 100 * float(normalized_data.loc[i, 'movement_the_day_after']) / float(normalized_data.loc[i, 'Open'])
        price_increase.append(increase)
    print("Collected evaluating the percentage of price increase in a day")

    classified_data = []
    for i in range(len(normalized_data)):
        if price_increase[i] <= -5:
            classified_data.append({"class": "huge_loss", "message": message[i]})
        if -5 < price_increase[i] <= -1:
            classified_data.append({"class": "loss", "message": message[i]})
        if -1 < price_increase[i] <= 1:
            classified_data.append({"class": "neutral", "message": message[i]})
        if 1 < price_increase[i] <= 5:
            classified_data.append({"class": "profit", "message": message[i]})
        if price_increase[i] > 5:
            classified_data.append({"class": "huge_profit", "message": message[i]})
    print("Training data was created")

    return classified_data


def classify_unique_words(training_data):
    corpus_words = {}
    class_words = {}

    classes = ["huge_loss", "loss", "neutral", "profit", "huge_profit"]
    for c in classes:
        class_words[c] = []

    for data in training_data:
        for word in nltk.word_tokenize(data['message']):
            if word not in ["?", "'s", ".", ","]:
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
        if word in class_words[class_name]:
            score += 1
    return score


def find_highest_class_score(message, class_words):
    for c in class_words.keys():
        score = calculate_class_score(message, class_words, c)
        print("Class: " + c + " Score: " + str(score) + "\n")


# Calculates the score of a message based both number of occurrences of a word in a class and the word commonality
def calculate_class_score_commonality(message, class_words, corpus_words, class_name, show_details=False):
    score = 1
    for word in nltk.word_tokenize(message):
        if word in class_words[class_name]:
            score *= class_words[class_name].count(word) / corpus_words[word]
            if show_details:
                print("   match: " + word + "(" + str(class_words[class_name].count(word) / corpus_words[word]) + ")")
        else:
            score *= 0.001
    return score


def find_highest_class_score_commonality(message, class_words, corpus_words):
    for c in class_words.keys():
        score = calculate_class_score_commonality(message, class_words, corpus_words, c)
        print("Class: " + c + " Score: " + str(score) + "\n")


def classify(message, class_words, corpus_words, show_details=False):
    high_class = None
    high_score = 0
    for c in class_words.keys():
        score = calculate_class_score_commonality(message, class_words, corpus_words, c, show_details)
        if score > high_score:
            high_class = c
            high_score = score

    return high_class, high_score


def split_data(data, test_size):
    training_data = data[:(len(data) - test_size)]
    test_data = data[-test_size:]

    return training_data, test_data


def train_model(data_path, test_size_percentage):
    data = pd.read_csv(data_path)
    classifying_data_time = datetime.now()
    print("Beginning classifying data: " + str(classifying_data_time))
    classified_data = build_classified_data(data)
    print("End of classifying data: " + str(datetime.now() - classifying_data_time))
    test_size = int(len(data) * test_size_percentage)
    train_data, test_data = split_data(classified_data, test_size)

    class_words, corpus_words = classify_unique_words(train_data)

    successful_predictions = 0
    for data in test_data:
        predicted_class, predicted_score = classify(data['message'], class_words, corpus_words)
        if data['class'] == predicted_class:
            successful_predictions += 1

    print("The size of the test data: " + str(test_size))
    print("Amount of successful predictions: " + str(successful_predictions))
    print("Accuracy achieved: " + str(100 * float(successful_predictions) / float(test_size)))

    return successful_predictions


def test_classification(data_path):
    data = pd.read_csv(data_path)
    data = build_classified_data(data)
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


def test_classification_predictions(data_path):
    begin_time = datetime.now()
    print("Start time: " + str(begin_time))
    train_model(data_path, 0.1)
    print("Time needed to train the model: ")
    print(datetime.now() - begin_time)
