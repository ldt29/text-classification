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
        self.embed_n_gram = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.fc1 = nn.Linear(embedding_size * 2, embedding_size)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(embedding_size, 5)
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
        embed_n_gram = self.embed_n_gram(inputs[1])
        outputs = torch.cat((embed_bow, embed_n_gram), -1)
        outputs = torch.mean(outputs, dim=1)
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        outputs = self.ReLU(outputs)
        outputs = self.fc2(outputs)
        outputs = self.softmax(outputs)
        #################################################################################################
        return outputs

    def init_weight(self, scope=1):
        self.embed_bow.weight.data.uniform_(-scope, scope)
        self.embed_n_gram.weight.data.uniform_(-scope, scope)
        self.fc1.weight.data.uniform_(-scope, scope)
        self.fc2.weight.data.uniform_(-scope, scope)



