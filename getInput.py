#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import nltk
from nltk.tokenize import WordPunctTokenizer
from path import *

word_cut = WordPunctTokenizer()
tokenizer = nltk.data.load(nltk_path)

dic_fir = open(all_path + "dic.pkl", "rb")
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
            temp.append(temp_words)
        datas.append(temp)
    f.close()
    return datas, labels, users, products



train_x, train_y, train_u, train_p = getInput(all_path + "data/IMDB/train.txt", 10)


for i in range(10):
    print(train_x[i])
# train_X = np.array(train_x)
# print("train_X.shape == ", train_X.shape)
# train_Y = np.array(train_y)
# print("train_Y == ", train_Y)
# print("train_Y.shape == ", train_Y.shape)
assert len(train_x) == len(train_y)
# indices = np.arange(train_X.shape[0])
# np.random.shuffle(indices)
# train_X = train_X[indices]
# train_Y = train_Y[indices]
print("len(train_x) = ", len(train_x))
train_fir = open(all_path + "train.pkl", "wb")
pickle.dump(train_x, train_fir)
pickle.dump(train_y, train_fir)
train_fir.close()




test_x, test_y, test_u, test_p = getInput(all_path + "data/IMDB/test.txt", 10)
# test_X = np.array(test_x)
# test_Y = np.array(test_y)
assert len(test_x) == len(test_y)
# indices = np.arange(test_X.shape[0])
# np.random.shuffle(indices)
# test_X = test_X[indices]
# test_Y = test_Y[indices]
print("len(test_x) = ", len(test_x))
test_fir = open(all_path + "test.pkl", "wb")
pickle.dump(test_x, test_fir)
pickle.dump(test_y, test_fir)
test_fir.close()