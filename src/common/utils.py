import re

from nltk import PorterStemmer, WordNetLemmatizer, TweetTokenizer
from nltk.corpus import stopwords


class Utils:
    BITCOIN_NAMES = ['BTC', 'btc', 'Btc', 'Bitcoin', 'bitcoin', 'bit']
    ETHEREUM_NAMES = ['ETH', 'eth', 'Eth', 'Ethereum', 'ETHEREUM', 'ethereum', 'ETHEREU']

    @staticmethod
    def clean_text(text, should_remove_signs):
        # Remove Unicode
        cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Remove Mentions
        cleaned_text = re.sub(r'@\w+', '', cleaned_text)
        # Remove the numbers
        # cleaned_text = re.sub(r'[0-9]', '', cleaned_text)
        # Remove the doubled space
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
        # Remove the doubled comma
        cleaned_text = re.sub(r',{2,}', ', ', cleaned_text)
        # Remove newlines and bad escape symbols
        cleaned_text = re.sub(r'\\u.{4}', '', cleaned_text)
        cleaned_text = re.sub(r'\\n', '', cleaned_text)
        # Remove unnecessary plus symbols
        cleaned_text = re.sub(r'\+', '', cleaned_text)
        cleaned_text = re.sub(r'#', '', cleaned_text)
        cleaned_text = re.sub(r'\(', '', cleaned_text)
        cleaned_text = re.sub(r'\)', '', cleaned_text)
        cleaned_text = re.sub(r':', '', cleaned_text)
        cleaned_text = re.sub(r';', '', cleaned_text)
        cleaned_text = re.sub(r'_', '', cleaned_text)
        cleaned_text = re.sub(r'\\', ' ', cleaned_text)
        cleaned_text = re.sub(r'-', ' ', cleaned_text)
        cleaned_text = re.sub(r'/', ' ', cleaned_text)
        cleaned_text = re.sub(r'\'', '', cleaned_text)
        cleaned_text = re.sub(r'\"', '', cleaned_text)
        cleaned_text = re.sub(r'\.([A-Za-z]{1})', r'. \1', cleaned_text)
        #cleaned_text = re.sub(r'https\:.* ', ' ', cleaned_text)

        if should_remove_signs:
            cleaned_text = re.sub(r'\?', ' ', cleaned_text)
            cleaned_text = re.sub(r'\.', ' ', cleaned_text)
            cleaned_text = re.sub(r'!', ' ', cleaned_text)

        return cleaned_text

    @staticmethod
    def normalize_text(text):
        text = Utils.clean_text(text, False)

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        normalized_text = ""
        tokenizer = TweetTokenizer()
        tokenized_text = tokenizer.tokenize(text)
        i = 0
        for word in tokenized_text:
            if word not in stop_words:
                lemmatized_word = lemmatizer.lemmatize(word)
                normalized_word = stemmer.stem(lemmatized_word.lower())
                # normalized_word = stemmer.stem(word.lower())
                normalized_text += normalized_word + (" " if i < len(tokenized_text) else "")
            i += 1

        return normalized_text

    @staticmethod
    def normalize_data(data):
        for i in range(len(data)):
            message = data.loc[i, 'message']
            if message and type(message) == str:
                normalized = Utils.normalize_text(message)
                data.loc[i, 'message'] = normalized

        print("Data was normalized")
        return data

    @staticmethod
    def normalize_data_to_list(data):
        normalized_data = []
        for i in range(len(data)):
            message = data.loc[i, 'message']
            if message and type(message) == str:
                normalized = Utils.normalize_text(message)
                normalized_data.append(normalized)

        return normalized_data
