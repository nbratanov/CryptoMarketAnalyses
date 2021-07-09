import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, n_filters, filter_sizes, pool_size, hidden_size,
                 number_of_classes, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_size)) for fs in filter_sizes])

        self.max_pool1 = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(95 * n_filters, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, number_of_classes, bias=True)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        convolution = [conv(embedded) for conv in self.convs]

        max1 = self.max_pool1(convolution[0].squeeze())
        max2 = self.max_pool1(convolution[1].squeeze())

        cat = torch.cat((max1, max2), dim=2)

        x = cat.view(cat.shape[0], -1)
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
