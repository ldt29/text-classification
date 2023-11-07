import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, hidden_size=512, num_layers=2, padding_idx=None):
        super(LSTM, self).__init__()
        #########################################  Your Code  ###########################################
        # todo
        # implement lstm
        # the output shape should be batch * 5

        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embed.weight.data.uniform_(-1, 1)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc1.weight.data.uniform_(-1, 1)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, 5)
        self.fc2.weight.data.uniform_(-1, 1)
        self.softmax = nn.Softmax(dim=1)
        #################################################################################################

    def forward(self, inputs, last_hidden=None):
        #inputs  Batch * seq_length

        #########################################  Your Code  ###########################################
        # todo
        # implement lstm
        # the output logits shape should be batch * 5

        # batch * seq_length * embedding_size
        embed = self.embed(inputs)
        # batch * seq_length * 2hidden_size
        _, (hidden, cell) = self.lstm(embed, last_hidden)
        # batch * hidden_size
        outputs = torch.cat((hidden[-2], hidden[-1]), dim=1)
        outputs = self.ReLU(outputs)
        outputs = self.dropout(outputs)
        # batch * 5
        outputs = self.fc1(outputs)
        outputs = self.ReLU(outputs)
        outputs = self.fc2(outputs)
        outputs = self.softmax(outputs)
        #################################################################################################
        return outputs

    




