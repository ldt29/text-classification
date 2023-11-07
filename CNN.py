import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, hidden_size=512, padding_idx=None):
        super(CNN, self).__init__()
        #########################################  Your Code  ###########################################
        # todo
        # implement text cnn
        # the output shape should be batch * 5
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.conv1 = nn.Conv1d(embedding_size, hidden_size, kernel_size=2)
        self.conv2 = nn.Conv1d(embedding_size, hidden_size, kernel_size=3)
        self.conv3 = nn.Conv1d(embedding_size, hidden_size, kernel_size=4)
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 5)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        #################################################################################################

    def forward(self, inputs):
        #inputs  Batch * seq_length

        #########################################  Your Code  ###########################################
        # todo
        # implement text cnn
        # the output logits shape should be batch * 5
        # batch * seq_length * embedding_size
        embed = self.embed(inputs)
        # batch * embedding_size * seq_length
        embed = embed.permute(0, 2, 1)
        # batch * hidden_size * (seq_length - kernel_size + 1)
        conv1 = self.conv1(embed)
        # batch * hidden_size * (seq_length - kernel_size + 1)
        conv2 = self.conv2(embed)
        # batch * hidden_size * (seq_length - kernel_size + 1)
        conv3 = self.conv3(embed)
        # batch * hidden_size 
        conv1 = torch.max(conv1, dim=-1)[0]
        conv2 = torch.max(conv2, dim=-1)[0]
        conv3 = torch.max(conv3, dim=-1)[0]
        outputs = torch.cat((conv1, conv2, conv3), -1)
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        outputs = self.ReLU(outputs)
        outputs = self.fc2(outputs)
        outputs = self.softmax(outputs)
        #################################################################################################
        return outputs





