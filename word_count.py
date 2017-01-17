import re
import heapq
import time
import numpy as np
from operator import itemgetter

vocab_path = "E:\LabWork\hdp\\test_data\\allWord.dat"
word_vect_path = "E:\LabWork\hdp\\test_data\\wordVect.dat"
result_file = file("E:\LabWork\hdp\\test_data\\word_count.dat", "w")

def word_count_noise(vocab_path, word_vect_path, result_file):
    vocab_list = [0]
    word_count = {}
    for line in open(vocab_path):
        vocab_list.append(line.strip('\n'))
    splitexp = re.compile(r'[ :]')
    ii = 0
    for line in open(word_vect_path):
        ii += 1
        # print "processing doc %d" % ii
        line = line.strip()
        if len(line) == 0:
            break
        splitline = [int(i) for i in splitexp.split(line)]
        wordids = splitline[1::2]
        wordcts = splitline[2::2]
        for wordid, wordct in zip(wordids, wordcts):
            if wordid not in word_count:
                word_count[wordid] = wordct
            else:
                word_count[wordid] += wordct
    result = sorted(word_count.iteritems(), key=lambda d: d[1], reverse=True)
    for item in result:
        result_file.write("%-10s:%-5d%-5s%d\n" % (vocab_list[item[0]], item[0], '-----', item[1]))

def word_count(vocab_path, word_vect_path, result_file):
    vocab_list = []
    word_count = {}
    flag = True
    for line in open(vocab_path):
        if flag:
            flag = False
            vocab_list = [0 for x in range(int(line) + 1)]
            continue
        split = line.strip("\n").rfind(':')
        vocab_list[int(line[split+1:])] = line[0:split]

    splitexp = re.compile(r'[ :]')

    ii = 0
    for line in open(word_vect_path):
        ii += 1
        print "processing doc %d" % ii
        line = line.strip()
        if len(line) == 0:
            break
        splitline = [int(i) for i in splitexp.split(line)]
        wordids = splitline[1::2]
        wordcts = splitline[2::2]
        for wordid, wordct in zip(wordids, wordcts):
            if wordid not in word_count:
                word_count[wordid] = wordct
            else:
                word_count[wordid] += wordct
    result = sorted(word_count.iteritems(), key=lambda d: d[1], reverse=True)
    for item in result:
        result_file.write("%-10s:%-5d%-5s%d\n" % (vocab_list[item[0]], item[0], '-----', item[1]))


def word_count_ap(vocab_path, word_vect_path, result_file):
    vocab_list = []
    word_count = {}
    vocab_list = str.split(file(vocab_path).read())
    splitexp = re.compile(r'[ :]')
    ii = 0
    for line in open(word_vect_path):
        ii += 1
        print "processing doc %d" % ii
        line = line.strip()
        if len(line) == 0:
            break
        splitline = [int(i) for i in splitexp.split(line)]
        wordids = splitline[1::2]
        wordcts = splitline[2::2]
        for wordid, wordct in zip(wordids, wordcts):
            if wordid not in word_count:
                word_count[wordid] = wordct
            else:
                word_count[wordid] += wordct
    result = sorted(word_count.iteritems(), key=lambda d: d[1], reverse=True)
    for item in result:
        result_file.write("%-10s:%-5d%-5s%d\n" % (vocab_list[item[0]], item[0], '-----', item[1]))

#word_count(vocab_path, word_vect_path, result_file)

vocab_path = "E:\LabWork\hdp\data\weibo_3M\\allWord_no_noise.dat"
word_vect_path = "E:\LabWork\hdp\data\weibo_3M\\wordVect_no_noise.dat"
result_file = file("E:\LabWork\hdp\data\weibo_3M\\weibo_count", "w")
word_count_noise(vocab_path, word_vect_path, result_file)