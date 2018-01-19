#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle


train_fir = open("/Users/ZRZn1/Downloads/train.pkl", "rb")
test_fir = open("/Users/ZRZn1/Downloads/test.pkl", "rb")
train_X = pickle.load(train_fir)
train_Y = pickle.load(train_fir)
test_X = pickle.load(test_fir)
test_Y = pickle.load(test_fir)

train_fir.close()
test_fir.close()

