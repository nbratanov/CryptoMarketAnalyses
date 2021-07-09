import os

import torch
from torch import nn
from torchtext import vocab

from src.sentiment_analysis.cnn import CNN
from src.sentiment_analysis.data_processing import get_data
from src.sentiment_analysis.train import create_iterator, run_train, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '/Users/bratanovn/Uni-Projects/CryptoMarketAnalyses'
path_data = os.path.join(path, "data")

learning_rate = 1e-4
batch_size = 50
dropout_keep_prob = 0.5
embedding_size = 300
max_message_length = 100
train_data_percentage = 0.8
max_size = 5000
seed = 1
num_classes = 3

training_data, validation_data, test_data, text, label = get_data(path_data, train_data_percentage, max_message_length,
                                                                  seed)

embeddings = vocab.Vectors('glove.840B.300d.txt', '../../data/glove_embedding/')
text.build_vocab(training_data, validation_data, max_size=max_size, vectors=embeddings)
label.build_vocab(training_data)
vocabulary_size = len(text.vocab)

training_iterator, validation_iterator, test_iterator = create_iterator(training_data, validation_data, test_data, batch_size,
                                                                        device)
loss_function = nn.CrossEntropyLoss()

hidden_size = 128
pool_size = 2
n_filters = 128
filter_sizes = [3, 8]
num_epochs = 5
should_train = True

cnn_model = CNN(vocabulary_size, embedding_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes,
                dropout_keep_prob)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

if should_train:
    run_train(num_epochs, cnn_model, training_iterator, validation_iterator, optimizer, loss_function)

cnn_model.load_state_dict(torch.load(os.path.join(path, "data/sentiment_analysis/saved_weights_CNN-2.pt")))

test_loss, test_accuracy = evaluate(cnn_model, test_iterator, loss_function)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_accuracy * 100:.2f}%')
