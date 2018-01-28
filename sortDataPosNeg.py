#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from path import *
import numpy as np

train_fir = open(all_path + "train.pkl", "rb")
test_fir = open(all_path + "test.pkl", "rb")
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

def sortBySen(s):
    max = 0
    for sen in s:
        if len(sen) > max:
            max = len(sen)
    return max

def getIndex(y):
    for i in range(len(y)):
        if y[i] == 1:
            return i
    return 0

for i in range(len(test_X)):
    if getIndex(test_Y[i]) < 6:
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
    if getIndex(train_Y[i]) < 6:
        neg_train_x.append(train_X[i])
        neg_train_y.append(train_Y[i])
        neg_train_u.append(train_U[i])
        neg_train_p.append(train_P[i])
    else:
        pos_train_x.append(train_X[i])
        pos_train_y.append(train_Y[i])
        pos_train_u.append(train_U[i])
        pos_train_p.append(train_P[i])


def sortData(x, y, u, p, count=256):
    assert len(x) == len(y)
    assert len(x) == len(u)
    assert len(x) == len(p)
    size = len(x)
    for i in range(size):
        y[i].append(u[i])
        y[i].append(p[i])
        x[i].append(y[i])

    x = sorted(x, key=lambda t: len(t))
    for i in range(size // count):
        temp = sorted(x[i * count: (i + 1) * count], key=lambda k: sortBySen(k))
        x[i * count: (i + 1) * count] = temp
    remain = size % count
    tempRemain = sorted(x[size - remain: size], key=lambda k: sortBySen(k))
    x[size - remain: size] = tempRemain

    # for i in range(10000, 11024):
    #     print(sortBySen(x[i]))

    res_y = []
    res_u = []
    res_p = []
    for i in range(size):
        temp_y = x[i].pop()
        res_p.append(temp_y.pop())
        res_u.append(temp_y.pop())
        res_y.append(temp_y)

    return x, res_y, res_u, res_p

# train_X, train_Y, train_U, train_P = sortData(train_X, train_Y, train_U, train_P, 512)
# test_X, test_Y, test_U, test_P = sortData(test_X, test_Y, test_U, test_P, 128)


pos_train_x, pos_train_y, pos_train_u, pos_train_p = sortData(pos_train_x, pos_train_y, pos_train_u, pos_train_p, 512)
neg_train_x, neg_train_y, neg_train_u, neg_train_p = sortData(neg_train_x, neg_train_y, neg_train_u, neg_train_p, 512)
pos_test_x, pos_test_y, pos_test_u, pos_test_p = sortData(pos_test_x, pos_test_y, pos_test_u, pos_test_p, 128)
neg_test_x, neg_test_y, neg_test_u, neg_test_p = sortData(neg_test_x, neg_test_y, neg_test_u, neg_test_p, 128)
# for i in range(15000, 16000):
#     print("U[" + str(i) + "] == ", pos_train_x[i])
#     print("P[" + str(i) + "] == ", pos_train_y[i])
# for i in range(22016, 22080):
#     print("len(train_X[" + str(i) +"]) == ", len(train_X[i]))
#
#
# for i in range(22014, 22080):
#     print("sortBySen[" + str(i) +"]) == ", sortBySen(train_X[i]))
# for l in range(len(train_X)-64, len(train_X)):
#     print("sortBySen[" + str(l) + "]) == ", sortBySen(train_X[l]))
#
# for l in range(10000, 11000):
#     print(train_Y[l])


def cutSame(x, count=64):
    size = len(x)
    for i in range(size // count):
        max_num = 0
        for j in range(i * count, (i + 1) * count):
            if len(x[j]) > max_num:
                max_num = len(x[j])
        max_len = sortBySen(x[(i + 1) * count - 1])
        if max_len > 50:
            max_len = sortBySen(x[(i + 1) * count - 2])
            if max_len > 75:
                max_len = sortBySen(x[(i + 1) * count - 3])
        zeroSen = []
        for t in range(max_len):
            zeroSen.append(0)
        for j in range(i * count, (i + 1) * count):
            for n in range(len(x[j])):
                if len(x[j][n]) < max_len:
                    for m in range(max_len - len(x[j][n])):
                        x[j][n].append(0)
                elif len(x[j][n]) > max_len:
                    x[j][n] = x[j][n][:max_len]
            if len(x[j]) < max_num:
                for m in range(max_num - len(x[j])):
                    x[j].append(zeroSen)
            elif len(x[j]) > max_num:
                print("这发生了！ == ", j)
                x[j] = x[j][:max_num]

    max_last_num = 0
    for i in range(size - size % count, size):
        if len(x[i]) > max_last_num:
            max_last_num = len(x[i])
    max_last_len = sortBySen(x[size - 2])
    zeroSen = []
    for t in range(max_last_len):
        zeroSen.append(0)
    for i in range(size - size % count, size):
        for n in range(len(x[i])):
            if len(x[i][n]) < max_last_len:
                for m in range(max_last_len - len(x[i][n])):
                    x[i][n].append(0)
            elif len(x[i][n]) > max_last_len:
                x[i][n] = x[i][n][:max_last_len]
        if len(x[i]) < max_last_num:
            for m in range(max_last_num - len(x[i])):
                x[i].append(zeroSen)
        elif len(x[i]) > max_last_num:
            print("这发生了！ == ", i)
            x[i] = x[i][:max_last_num]
    return x


pos_train_x = cutSame(pos_train_x)
neg_train_x = cutSame(neg_train_x)
pos_test_x = cutSame(pos_test_x)
neg_test_x = cutSame(neg_test_x)

# def changeY(y, isPos):
#     if not isPos:
#         for i in range(len(y)):
#             y[i] = y[i][:6]
#     else:
#         for i in range(len(y)):
#             y[i] = y[i][6:]
#     return y
#
# pos_train_y = changeY(pos_train_y, True)
# neg_train_y = changeY(neg_train_y, False)
# pos_test_y = changeY(pos_test_y, True)
# neg_test_y = changeY(neg_test_y, False)
#
# for i in range(1000, 1200):
#     print(neg_train_y[i])


train_out_np = open(all_path + "train_out_np.pkl", "wb")
test_out_np = open(all_path + "test_out_np.pkl", "wb")

pickle.dump(pos_train_x, train_out_np)
pickle.dump(pos_train_y, train_out_np)
pickle.dump(pos_train_u, train_out_np)
pickle.dump(pos_train_p, train_out_np)
pickle.dump(neg_train_x, train_out_np)
pickle.dump(neg_train_y, train_out_np)
pickle.dump(neg_train_u, train_out_np)
pickle.dump(neg_train_p, train_out_np)
train_out_np.close()

pickle.dump(pos_test_x, test_out_np)
pickle.dump(pos_test_y, test_out_np)
pickle.dump(pos_test_u, test_out_np)
pickle.dump(pos_test_p, test_out_np)
pickle.dump(neg_test_x, test_out_np)
pickle.dump(neg_test_y, test_out_np)
pickle.dump(neg_test_u, test_out_np)
pickle.dump(neg_test_p, test_out_np)
test_out_np.close()