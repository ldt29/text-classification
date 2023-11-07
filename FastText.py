import torch
import torch.nn as nn

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_size=512, padding_idx=None):
        super(FastText, self).__init__()
        #########################################  Your Code  ###########################################
        # todo
        # implement fast text
        # the output shape should be batch * 5
        self.embed_bow = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embed_bow.weight.data.uniform_(-0.1, 0.1)
        self.embed_bigram = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embed_bigram.weight.data.uniform_(-0.1, 0.1)
        self.fc1 = nn.Linear(embedding_size * 2, embedding_size)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(embedding_size, 5)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        #################################################################################################

    def forward(self, inputs):
        #inputs  Batch * seq_length

        #########################################  Your Code  ###########################################
        # todo
        # implement fast text
        # the output logits shape should be batch * 5
        embed_bow = self.embed_bow(inputs[0])
        embed_bigram = self.embed_bigram(inputs[1])
        outputs = torch.cat((embed_bow, embed_bigram), -1)
        outputs = torch.mean(outputs, dim=1)
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        outputs = self.ReLU(outputs)
        outputs = self.fc2(outputs)
        outputs = self.softmax(outputs)
        #################################################################################################
        return outputs






