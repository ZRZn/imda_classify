#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from path import *
import numpy as np

train_fir = open(all_path + "train.pkl", "rb")
test_fir = open(all_path + "test.pkl", "rb")
train_X = pickle.load(train_fir)
train_Y = pickle.load(train_fir)
test_X = pickle.load(test_fir)
test_Y = pickle.load(test_fir)
train_fir.close()
test_fir.close()


def sortBySen(s):
    max = 0
    for sen in s:
        if len(sen) > max:
            max = len(sen)
    return max


def sortData(x, y, count=256):
    assert len(x) == len(y)
    size = len(x)
    for i in range(size):
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
    for i in range(size):
        res_y.append(x[i].pop())

    return x, res_y

train_X, train_Y = sortData(train_X, train_Y, 512)
test_X, test_Y = sortData(test_X, test_Y, 128)
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
    max_last_len = sortBySen(x[size - 3])
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

train_X = cutSame(train_X)
test_X = cutSame(test_X)


train_out = open(all_path + "train_out.pkl", "wb")
test_out = open(all_path + "test_out.pkl", "wb")

pickle.dump(train_X, train_out)
pickle.dump(train_Y, train_out)
train_out.close()

pickle.dump(test_X, test_out)
pickle.dump(test_Y, test_out)
test_out.close()