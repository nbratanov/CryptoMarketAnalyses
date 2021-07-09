import pandas as pd
import re
import nltk
from src.common.utils import Utils

from nltk.corpus import stopwords

dataset = pd.read_csv('../data/fullData.csv')

#
# nltk.download('stopwords')
# s = stopwords.words('english')
# ps = nltk.wordnet.WordNetLemmatizer()
#
# for i in range(len(dataset)):
#     message = re.sub('[^a-zA-Z]', ' ', dataset.loc[i,'message'])
#     message = message.lower()
#     message = message.split()
#     message = [ps.lemmatize(word) for word in message if not word in s]
#     message = ' '.join(message)
#     dataset.loc[i, 'message'] = message

#dataset2 = dataset['message'].apply(lambda x: len(x.split(' '))>2)
def replace_bitcoin_mentions(message):
    words = message.split(' ')
    for i in range(len(words)):
        if words[i] in Utils.BITCOIN_NAMES:
            words[i] = 'Bitcoin'
    return " ".join(words)

dataset = dataset[dataset['message'].apply(lambda x: len(x.split(' ')) > 2)]
dataset = dataset[~dataset['message'].apply(lambda x: 'https' in x)]
dataset['message'] = dataset['message'].apply(lambda x: Utils.clean_text(x, True))
dataset['message'] = dataset['message'].apply(lambda x: replace_bitcoin_mentions(x))
dataset['message'] = dataset['message'].apply(lambda x: Utils.normalize_text(x))


dataset.to_csv('../data/fullData_normalized_withoutLinks.csv')

#
# for index, row in dataset.iterrows():
#     message = row['message']
#     if(len(message.split(' ')) <= 2):
#         count += 1
# print(count)