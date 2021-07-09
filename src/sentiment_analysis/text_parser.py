import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

data = pd.read_csv('../../data/bitcoin-data.csv')
analyser = SentimentIntensityAnalyzer()
scores = []
for message in data['message']:
    try:
        score = analyser.polarity_scores(message)
        scores.append(score)
    except AttributeError as e:
        raise AttributeError("Could not parse data on message: " + str(message))

new_data = pd.DataFrame(columns=['text', 'label'])

for i in range(len(data)):
    score = 0
    if scores[i]['compound'] >= 0.05:
        score = 0
    elif scores[i]['compound'] <= -0.05:
        score = 1
    else:
        score = 2
    new_data.loc[i, ['text']] = data.loc[i, 'message']
    new_data.loc[i, ['label']] = score

training_data, test_data = train_test_split(new_data, test_size=0.2)

new_data.to_csv('../../data/sentiment_analysis/complete_data-full-btc.csv')
training_data.to_csv('../../data/sentiment_analysis/training_data-full-btc.csv')
test_data.to_csv('../../data/sentiment_analysis/test_data-full-btc.csv')
