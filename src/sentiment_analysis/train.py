import matplotlib.pyplot as plt
import torch
from torchtext.legacy import data


def accuracy(probs, target):
    winners = probs.argmax(dim=1)
    corrects = (winners == target)
    accuracy = corrects.sum().float() / float(target.size(0))
    return accuracy


def create_iterator(train_data, valid_data, test_data, batch_size, device):
    training_iterator, validation_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                               batch_size=batch_size,
                                                                               sort_key=lambda x: len(x.text),
                                                                               sort_within_batch=True, device=device)
    return training_iterator, validation_iterator, test_iterator


def train(cnn_model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text

        predictions = cnn_model(text, text_lengths)
        loss = criterion(predictions, batch.label.squeeze())

        acc = accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(cnn_model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    cnn_model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = cnn_model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = acc(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def run_train(epochs, cnn_model, training_iterator, validation_iterator, optimizer, criterion):
    best_validation_loss = float('inf')

    for epoch in range(epochs):

        training_loss, training_accuracy = train(cnn_model, training_iterator, optimizer, criterion)
        validation_loss, validation_accuracy = evaluate(cnn_model, validation_iterator, criterion)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(cnn_model.state_dict(), '../../data/sentiment_analysis/saved_weights_CNN-2.pt')

        print(f'\tTrain Loss: {training_loss:.3f} | Train Acc: {training_accuracy * 100:.2f}%')
        print(f'\t Val. Loss: {validation_loss:.3f} |  Val. Acc: {validation_accuracy * 100:.2f}%')
