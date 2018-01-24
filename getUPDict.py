#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from path import *
import pickle

def getDict(file_path, dict_name='temp.dict'):
    f = open(file_path, 'rb')
    data = dict()
    for line in f:
        print(line)
        line = line[1:len(line) - 1]
        line = "\\" + line
        data[line] = len(data)
    return data


usr_dict = getDict(all_path + "/data/IMDB/prdlist.txt")
# print usr_dict
# print "长度为：", len(usr_dict)
# t = '\\tt0047472'
# print "\tt0047472 == ", usr_dict[t]
# usr_file = open(all_path + "prd.dict", 'wb')
# pickle.dump(usr_dict, usr_file)



# usr_file = open(all_path + "usr.dict", 'rb')
# usr_dict = pickle.load(usr_file)
# print(usr_dict)
# print(type(usr_dict))
# t = 'ur1577474/'
# print("ur1517556/ == ", usr_dict[t])

# print(usr_dict)
# print("ur1517556/ == ", usr_dict['ur1577474/'])