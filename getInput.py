#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import nltk
from nltk.tokenize import WordPunctTokenizer

word_cut = WordPunctTokenizer()
tokenizer = nltk.data.load('/Users/zrzn/Downloads/nltk_data/tokenizers/punkt/PY3/english.pickle')

dic_fir = open("/Users/zrzn/Downloads/classify_data/dic.pkl", "rb")
dictionary = pickle.load(dic_fir)
dic_fir.close()
print(type(dictionary))
print(dictionary['the'])

def getInput(file_path, classify=10):
    f = open(file_path, "r")
    datas = []
    labels = []
    users = []
    products = []
    zeroSen = []
    for i in range(30):
        zeroSen.append(0)
    for line in f:
        line = line.replace('<sssss>', '')
        line = line.split("		")
        users.append(line[0])
        products.append(line[1])
        label = int(line[2])
        tempLabel = []
        for i in range(1, classify + 1):
            if i == label:
                tempLabel.append(1)
            else:
                tempLabel.append(0)
        labels.append(tempLabel)
        sentences = tokenizer.tokenize(line[3])
        temp = []
        for sentence in sentences:
            if len(sentence) == 0:
                continue
            words = word_cut.tokenize(sentence)
            temp_words = []
            for word in words:
                temp_index = 0
                if word in dictionary:
                    temp_index = dictionary[word]
                temp_words.append(temp_index)
            if len(temp_words) < 30:
                for i in range(30 - len(temp_words)):
                    temp_words.append(0)
            elif len(temp_words) > 30:
                temp_words = temp_words[:30]
            temp.append(temp_words)
        if len(temp) < 11:
            for i in range(11 - len(temp)):
                temp.append(zeroSen)
        elif len(temp) > 11:
            temp = temp[:11]
        datas.append(temp)
    f.close()
    return datas, labels, users, products



train_x, train_y, train_u, train_p = getInput("/Users/zrzn/Downloads/data/IMDB/train.txt", 10)


for i in range(10):
    print(train_x[i])
train_X = np.array(train_x)
print("train_X.shape == ", train_X.shape)
train_Y = np.array(train_y)
print("train_Y == ", train_Y)
print("train_Y.shape == ", train_Y.shape)
assert train_X.shape[0] == train_Y.shape[0]
indices = np.arange(train_X.shape[0])
np.random.shuffle(indices)
train_X = train_X[indices]
train_Y = train_Y[indices]
train_fir = open("/Users/zrzn/Downloads/classify_data/train.pkl", "wb")
pickle.dump(train_X, train_fir)
pickle.dump(train_Y, train_fir)
train_fir.close()




test_x, test_y, test_u, test_p = getInput("/Users/zrzn/Downloads/data/IMDB/test.txt", 10)
test_X = np.array(test_x)
test_Y = np.array(test_y)
assert test_X.shape[0] == test_Y.shape[0]
indices = np.arange(test_X.shape[0])
np.random.shuffle(indices)
test_X = test_X[indices]
test_Y = test_Y[indices]
test_fir = open("/Users/zrzn/Downloads/classify_data/test.pkl", "wb")
pickle.dump(test_X, test_fir)
pickle.dump(test_Y, test_fir)
test_fir.close()