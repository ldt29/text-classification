import json
import os
import copy
import re
from torch.utils.data import Dataset, DataLoader
import torch
import sys
from dictionary import Dictionary
import csv
import collections
import nltk
import logging
import random
import numpy as np
from collections import defaultdict

def collate_fn(batch):
    max_length = 0
    for token_id, label, length in batch:
        max_length = max(length, max_length)
    token_ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, (token_id, label, length) in enumerate(batch):
        token_ids[i] = torch.cat([token_id, torch.tensor([0] * (max_length - len(token_id)), dtype=torch.long)])
        labels[i] = label
    return token_ids, labels

def collate_fn_fasttext(batch):
    max_length = 0
    for token_id, n_gram, label, length in batch:
        max_length = max(length, max_length)
    token_ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    n_grams = torch.zeros(len(batch), max_length, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, (token_id, n_gram, label, length) in enumerate(batch):
        token_ids[i] = torch.cat([token_id, torch.tensor([0] * (max_length - len(token_id)), dtype=torch.long)])
        n_grams[i] = torch.cat([n_gram, torch.tensor([0] * (max_length - len(n_gram)), dtype=torch.long)])
        labels[i] = label
    return torch.stack([token_ids, n_grams]), labels

class CLSDataset(Dataset):
    def __init__(self, data_path="./yelp_small/", dictionary=None, split='train', block_size=0):

        self.filename = os.path.join(data_path, "{}.csv".format(split))
        self.data = []
        self.dictionary = dictionary
        self.padding_idx = self.dictionary.pad()
        self.vocab_size = len(dictionary)
        self.block_size = block_size
        self.max_length = 0
        self.lens = []

        with open(self.filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                label = int(row[0])
                raw_text = row[1]
                tokens = row[1].replace('\\n', ' ')
                tokens = re.sub(r'[^a-zA-Z\s]', ' ', string=tokens)
                tokens = tokens.strip().split(' ')
                
                for i in reversed(range(len(tokens))):
                    if tokens[i] == '': 
                        assert tokens.pop(i) == ''
                        continue

                    tokens[i] = tokens[i].lower()
                    if tokens[i] not in self.dictionary.indices.keys():
                        tokens[i] = '<unk>'
                token_id = self.dictionary.encode_line(tokens)
                self.max_length = max(self.max_length, len(tokens))
                self.lens.append(len(tokens))
                self.data.append((raw_text, tokens, token_id, label, len(token_id)))
        self.data = sorted(self.data, key = lambda x:x[4])
        self.idx = list(range(len(self.data)))
        if self.block_size > 0:
            l = 0
            while (l < len(self.idx)):
                r = min(len(self.idx), l + self.block_size)
                tmp = self.idx[l:r]
                random.shuffle(tmp)
                self.idx[l:r] = tmp
                l = r
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''return a batch of samples'''
        raw_text, tokens, token_id, label, length = self.data[self.idx[index]] 
        return token_id, label, length


class BOWDataset(Dataset):
    def __init__(self, data_path="./yelp_small/", dictionary=None, split='train', n=0):
        # n for n_gram
        self.filename = os.path.join(data_path, "{}.csv".format(split))
        self.data = []
        self.dictionary = dictionary
        self.vocab_size = len(dictionary) 
        self.stopwords = nltk.corpus.stopwords.words('english')

        #########################################  Your Code  ###########################################
        # todo
        with open(self.filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                label = int(row[0])
                raw_text = row[1]
                tokens = row[1].replace('\\n', ' ')
                # Remove all the tokens that contain numbers.
                tokens = re.sub(r'\w*\d\w*', ' ', string=tokens)
                # Remove all the punctuation in the tokens.
                tokens = re.sub(r'[\']', '', string=tokens)
                tokens = re.sub(r'[^a-zA-Z\s]', ' ', string=tokens)
                tokens = tokens.strip().split(' ')
                
                for i in reversed(range(len(tokens))):
                    if tokens[i] == '': 
                        assert tokens.pop(i) == ''
                        continue
                    # Remove the stop words first.
                    tokens[i] = tokens[i].lower()
                    if tokens[i] in self.stopwords:
                        assert tokens.pop(i) in self.stopwords
                        continue
                    
                    if tokens[i] not in self.dictionary.indices.keys():
                        tokens[i] = '<unk>'
                token_id = self.dictionary.encode_line(tokens)
                bow_freq_feature = self.get_bow_freq_feature(token_id, n)
                self.data.append((raw_text, tokens, token_id, bow_freq_feature, label))
        self.data = sorted(self.data, key = lambda x:x[4])
        self.idx = list(range(len(self.data)))
        #################################################################################################
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # bow_freq_feature and label
        return self.data[index][-2:]

    def get_bow_freq_feature(self, token_id, n):
        bow_freq_feature = torch.zeros(self.vocab_size)
        if n == 0:
            for i in token_id:
                bow_freq_feature[i] += 1
        else:
            for i in range(len(token_id) - n + 1):
                bow_freq_feature[token_id[i:i+n]] += 1
        return bow_freq_feature

class FastTextDataset(Dataset):
    def __init__(self, data_path="./yelp_small/", dictionary=None, split='train', n=2):

        self.filename = os.path.join(data_path, "{}.csv".format(split))
        self.data = []
        self.dictionary = dictionary
        self.padding_idx = self.dictionary.pad()
        self.vocab_size = len(dictionary)
        self.total_size = 1007
        self.lens = []

        #########################################  Your Code  ###########################################
        # todo
        with open(self.filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                label = int(row[0])
                raw_text = row[1]
                tokens = row[1].replace('\\n', ' ')
                # Remove all the punctuation in the tokens.
                tokens = re.sub(r'[^a-zA-Z\s]', ' ', string=tokens)
                tokens = tokens.strip().split(' ')
                
                for i in reversed(range(len(tokens))):
                    if tokens[i] == '': 
                        assert tokens.pop(i) == ''
                        continue

                    tokens[i] = tokens[i].lower()
                    if tokens[i] not in self.dictionary.indices.keys():
                        tokens[i] = '<unk>'
                token_id = self.dictionary.encode_line(tokens)
                n_grams = nltk.ngrams(token_id, n)
                word_hash = self.wordhash(token_id)
                n_gram_hash = self.n_gramhash(n_grams, n)
                self.lens.append(len(token_id))
                self.data.append((raw_text, tokens, word_hash, n_gram_hash, label, len(token_id)))
        #################################################################################################

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_text, tokens, token_id, n_gram , label, length = self.data[index] 
        return token_id, n_gram, label, length

    
    def wordhash(self, token_id):
        return torch.tensor([token_id[i]*14918087 % self.total_size for i in range(len(token_id))])
    
    def n_gramhash(self, n_grams, n):
        if n == 2:
            return torch.tensor([(tokens[0]*14918087*18408749 + tokens[1]*14918087) % self.total_size for tokens in n_grams])
        if n == 3:
            return torch.tensor([(tokens[0]*14918087*18408749*5971847 + tokens[1]*14918087*18408749 + tokens[2]*14918087) % self.total_size for tokens in n_grams])
        else:
            raise NotImplementedError 
    
    

        

        
        
