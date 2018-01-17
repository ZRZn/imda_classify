#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

dic_fir = open("/Users/ZRZn1/Downloads/dic.pkl", "rb")
dictionary = pickle.load(dic_fir)
dic_fir.close()
print(type(dictionary))
print(dictionary['the'])

def getInput(file_path, classify=0):
    dirs = os.listdir(file_path)
    data = []
    labels = []
    zeroSen = []
    for i in range(40):
        zeroSen.append(0)
    for dir in dirs:
        f = open(file_path + "/" + dir, "r")
        temp = []
        for line in f:
            line = line.replace('!', '.')
            line = line.replace('<br /><br />', '')
            line = line.replace('?', '.')
            line = line.replace('*', '')
            line = line.replace(',', '')
            sentences = line.split('.')
            for sentence in sentences:
                if len(sentence) == 0:
                    continue
                words = sentence.split()
                temp_words = []
                for word in words:
                    temp_index = 0
                    if word in dictionary:
                        temp_index = dictionary[word]
                    temp_words.append(temp_index)
                if len(temp_words) < 40:
                    for i in range(40 - len(temp_words)):
                        temp_words.append(0)
                elif len(temp_words) > 40:
                    temp_words = temp_words[:40]
                temp.append(temp_words)
        if len(temp) < 10:
            for i in range(10 - len(temp)):
                temp.append(zeroSen)
        elif len(temp) > 10:
            temp = temp[:10]
        data.append(temp)
        labels.append(classify)
        f.close()
    return data, labels


train_pos, train_label_pos = getInput("/Users/ZRZn1/Downloads/aclImdb/train/pos", 1)
train_neg, train_label_neg = getInput("/Users/ZRZn1/Downloads/aclImdb/train/neg", 0)


for train in train_neg:
    train_pos.append(train)
for y in train_label_neg:
    train_label_pos.append(y)
train_X = np.array(train_pos)
count = 0
for train in train_X:
    count += 1
print("count == " + str(count))
train_Y = np.array(train_label_pos)
print("train_Y == ", train_Y)
print("train_Y.shape == ", train_Y.shape)
assert train_X.shape[0] == train_Y.shape[0]
indices = np.arange(train_X.shape[0])
np.random.shuffle(indices)
train_X = train_X[indices]
train_Y = train_Y[indices]
print("train_X.shape == " + str(train_X.shape))
train_fir = open("/Users/ZRZn1/Downloads/train.pkl", "wb")
pickle.dump(train_X, train_fir)
pickle.dump(train_Y, train_fir)
train_fir.close()




test_pos, test_label_pos = getInput("/Users/ZRZn1/Downloads/aclImdb/test/pos", 1)
test_neg, test_label_neg = getInput("/Users/ZRZn1/Downloads/aclImdb/test/neg", 0)

for test in test_neg:
    test_pos.append(test)
for y in test_label_neg:
    test_label_pos.append(y)
test_X = np.array(test_pos)
test_Y = np.array(test_label_pos)
assert test_X.shape[0] == test_Y.shape[0]
indices = np.arange(test_X.shape[0])
np.random.shuffle(indices)
test_X = test_X[indices]
test_Y = test_Y[indices]
test_fir = open("/Users/ZRZn1/Downloads/test.pkl", "wb")
pickle.dump(test_X, test_fir)
pickle.dump(test_Y, test_fir)
test_fir.close()