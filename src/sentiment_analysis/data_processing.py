import random

import spacy
from spacy.tokenizer import Tokenizer
from torchtext.legacy import data

from src.common.utils import Utils


def get_data(path, train_data_percentage, max_message_length, seed):
    # nlp = spacy.load("en_core_web_sm")
    # tokenizer = Tokenizer(nlp.vocab)
    tokenizer = lambda s: s.split()
    text = data.Field(preprocessing=Utils.clean_tokenized_text, tokenize=tokenizer, batch_first=True,
                      include_lengths=True, fix_length=max_message_length)
    label = data.Field(sequential=False, use_vocab=False, pad_token=False, unk_token=None)

    fields = [('id', None), ('text', text), ('label', label)]
    training_data, test_data = data.TabularDataset.splits(
        path=path,
        train='./sentiment_analysis/training_data-full-btc.csv',
        test='./sentiment_analysis/test_data-full-btc.csv',
        format='csv',
        fields=fields,
        skip_header=True
    )

    training_data, validation_data = training_data.split(split_ratio=train_data_percentage,
                                                         random_state=random.seed(seed))

    print(vars(training_data[0]))
    print(f'Number of training examples: {len(training_data)}')
    print(f'Number of validation examples: {len(validation_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    return training_data, validation_data, test_data, text, label
