#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from path import *
import pickle


#Load Data
train_fir = open(all_path + "train_out.pkl", "rb")
test_fir = open(all_path + "test_out.pkl", "rb")
train_X = pickle.load(train_fir)
train_Y = pickle.load(train_fir)
train_U = pickle.load(train_fir)
train_P = pickle.load(train_fir)
test_X = pickle.load(test_fir)
test_Y = pickle.load(test_fir)
test_U = pickle.load(test_fir)
test_P = pickle.load(test_fir)

train_fir.close()
test_fir.close()

pos_train_x = []
pos_train_y = []
pos_train_u = []
pos_train_p = []
neg_train_x = []
neg_train_y = []
neg_train_u = []
neg_train_p = []


pos_test_x = []
pos_test_y = []
pos_test_u = []
pos_test_p = []
neg_test_x = []
neg_test_y = []
neg_test_u = []
neg_test_p = []



def getIndex(y):
    for i in range(len(y)):
        if y[i] == 1:
            return i
    return 0

for i in range(len(test_X)):
    if getIndex(test_Y[i]) < 5:
        neg_test_x.append(test_X[i])
        neg_test_y.append(test_Y[i])
        neg_test_u.append(test_U[i])
        neg_test_p.append(test_P[i])
    else:
        pos_test_x.append(test_X[i])
        pos_test_y.append(test_Y[i])
        pos_test_u.append(test_U[i])
        pos_test_p.append(test_P[i])


for i in range(len(train_X)):
    if getIndex(train_Y[i]) < 5:
        neg_train_x.append(train_X[i])
        neg_train_y.append(train_Y[i])
        neg_train_u.append(train_U[i])
        neg_train_p.append(train_P[i])
    else:
        pos_train_x.append(train_X[i])
        pos_train_y.append(train_Y[i])
        pos_train_u.append(train_U[i])
        pos_train_p.append(train_P[i])

for i in range(11200, 11520):
    print(len(pos_train_x[i]))