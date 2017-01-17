#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def str_has_non_chinese(ustring):
    """判断字符串是否有非中文字符"""
    for uchar in ustring.decode('utf-8'):
        if not is_chinese(uchar):
            return True
    return False


def remove_noise_with_time_location(input_filename, output_file, noise_dict, rearrange_vocab):
    if not os.path.exists(input_filename):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % input_filename
    splitexp = re.compile(r'[ :\t]')
    for line in file(input_filename):
        splitline = [int(i) for i in splitexp.split(line)[:-8]]
        extrainfo = splitexp.split(line)[-8:]
        doc_words = splitline[1::2]
        doc_counts = splitline[2::2]
        flag = True  # 表示这一行没有被去掉
        count = 0
        for i in range(len(doc_words)):
            if doc_words[i] in noise_dict:
                doc_words[i] = 0
                count += 1
            else:
                doc_words[i] = rearrange_vocab[doc_words[i]]
        if count == len(doc_words):
            flag = False
        if flag and len(doc_words) != 0:
            output_file.write('%d ' % (len(doc_words) - count))
            for i in range(len(doc_words)):
                if doc_words[i] != 0:
                    output_file.write('%d:%d ' % (doc_words[i], doc_counts[i]))
            output_file.write('\t%s %s %s:%s:%s %s %s %s' %
                              (extrainfo[0], extrainfo[1], extrainfo[2], extrainfo[3], extrainfo[4], extrainfo[5],
                               extrainfo[6], extrainfo[7]))
            # output_file.write('\n')
    print "finished remove noise."


def remove_noise(input_filename, output_file, noise_dict, rearrange_vocab):
    if not os.path.exists(input_filename):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % input_filename
    splitexp = re.compile(r'[ :]')
    for line in file(input_filename):
        splitline = [int(i) for i in splitexp.split(line)]
        doc_words = splitline[1::2]
        doc_counts = splitline[2::2]
        flag = True  # 表示这一行没有被去掉
        count = 0
        for i in range(len(doc_words)):
            if doc_words[i] in noise_dict:
                doc_words[i] = 0
                count += 1
            else:
                doc_words[i] = rearrange_vocab[doc_words[i]]
        if count == len(doc_words):
            flag = False
        if flag and len(doc_words) != 0:
            output_file.write('%d ' % (len(doc_words) - count))
            for i in range(len(doc_words)):
                if doc_words[i] != 0:
                    output_file.write('%d:%d ' % (doc_words[i], doc_counts[i]))
            output_file.write('\n')
    print "finished remove noise."


def remove_too_short_tweet(input_filename, output_file, threshold):
    if not os.path.exists(input_filename):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % input_filename
    splitexp = re.compile(r'[ :]')
    for line in file(input_filename):
        line = line.strip()
        splitline = [int(i) for i in splitexp.split(line)]
        doc_words = splitline[1::2]
        if len(doc_words) >= threshold:
            output_file.write(line + "\n")


def remove_too_short_tweet_with_time_location(input_filename, output_file, threshold):
    if not os.path.exists(input_filename):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % input_filename
    splitexp = re.compile(r'[ :\t]')
    for line in file(input_filename):
        line = line.strip()
        splitline = [int(i) for i in splitexp.split(line)[:-9]]
        # extrainfo = splitexp.split(line)[-8:]
        doc_words = splitline[1::2]
        if len(doc_words) >= threshold:
            output_file.write(line+'\n')
            # output_file.write('\t%s %s %s:%s:%s %s %s %s' %
            #                   (extrainfo[0], extrainfo[1], extrainfo[2], extrainfo[3], extrainfo[4], extrainfo[5],
            #                    extrainfo[6], extrainfo[7]))


def remove_common_word(input_filename, output_file, threshold):
    if not os.path.exists(input_filename):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % input_filename
    splitexp = re.compile(r'[ :]')
    apperance_dict = {}
    doc_counts = 0
    for line in file(input_filename):
        doc_counts += 1
        line = line.strip()
        splitline = [int(i) for i in splitexp.split(line)]
        doc_words = splitline[1::2]
        doc_counts = splitline[2::2]
        for i in range(len(doc_words)):
            if doc_words[i] not in apperance_dict:
                apperance_dict[doc_words[i]] = doc_counts[i]
            else:
                apperance_dict[doc_words[i]] += doc_counts[i]
    for item in apperance_dict.keys():
        if apperance_dict[item] <= threshold:
            del apperance_dict[item]
    # print apperance_dict

    # 去掉里面的常见词
    for line in file(input_filename):
        splitline = [int(i) for i in splitexp.split(line)]
        doc_words = splitline[1::2]
        doc_counts = splitline[2::2]
        flag = True  # 表示这一行没有被去掉
        count = 0
        for i in range(len(doc_words)):
            if doc_words[i] in apperance_dict:
                doc_words[i] = 0
                count += 1
        if count == len(doc_words):
            flag = False
        if flag and len(doc_words) != 0:
            output_file.write('%d ' % (len(doc_words) - count))
            for i in range(len(doc_words)):
                if doc_words[i] != 0:
                    output_file.write('%d:%d ' % (doc_words[i], doc_counts[i]))
            output_file.write('\n')


            # print "finished remove common words."


def remove_too_common_word_with_time_location(input_filename, output_file, too_common_dict, rearrange_vocab):
    if not os.path.exists(input_filename):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % input_filename
    splitexp = re.compile(r'[ :\t]')
    for line in file(input_filename):
        line = line.strip()
        splitline = [int(i) for i in splitexp.split(line)[:-9]]
        extrainfo = splitexp.split(line)[-8:]
        doc_words = splitline[1::2]
        doc_counts = splitline[2::2]
        flag = True  # 表示这一行没有被去掉
        count = 0
        for i in range(len(doc_words)):
            if doc_words[i] in too_common_dict:
                doc_words[i] = 0
                count += 1
            else:
                doc_words[i] = rearrange_vocab[doc_words[i]]
        if count == len(doc_words):
            flag = False
        if flag and len(doc_words) != 0:
            output_file.write('%d ' % (len(doc_words) - count))
            for i in range(len(doc_words)):
                if doc_words[i] != 0:
                    output_file.write('%d:%d ' % (doc_words[i], doc_counts[i]))
            output_file.write('\t%s %s %s:%s:%s %s %s %s\n' %
                              (extrainfo[0], extrainfo[1], extrainfo[2], extrainfo[3], extrainfo[4], extrainfo[5],
                               extrainfo[6], extrainfo[7]))
            # print '\t%s %s %s:%s:%s %s %s %s' % (extrainfo[0], extrainfo[1], extrainfo[2], extrainfo[3], extrainfo[4], extrainfo[5],
            #                    extrainfo[6], extrainfo[7])
            # output_file.write('\n')
    print "finished remove too common words."


def remove_too_common_word(input_filename, output_file, too_common_dict, rearrange_vocab):
    if not os.path.exists(input_filename):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % input_filename
    splitexp = re.compile(r'[ :]')
    for line in file(input_filename):
        line = line.strip()
        splitline = [int(i) for i in splitexp.split(line)]
        doc_words = splitline[1::2]
        doc_counts = splitline[2::2]
        flag = True  # 表示这一行没有被去掉
        count = 0
        for i in range(len(doc_words)):
            if doc_words[i] in too_common_dict:
                doc_words[i] = 0
                count += 1
            else:
                doc_words[i] = rearrange_vocab[doc_words[i]]
        if count == len(doc_words):
            flag = False
        if flag and len(doc_words) != 0:
            output_file.write('%d ' % (len(doc_words) - count))
            for i in range(len(doc_words)):
                if doc_words[i] != 0:
                    output_file.write('%d:%d ' % (doc_words[i], doc_counts[i]))
            output_file.write('\n')
    print "finished remove too common words."


def collect_too_common_word_with_time_location(vocab_path, vect_path, output_vocab, threshold):
    if not os.path.exists(vect_path):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % vect_path
    splitexp = re.compile(r'[ :\t]')
    apperance_dict = {}
    for line in file(vect_path):
        line = line.strip()
        # print splitexp.split(line)
        splitline = [int(i) for i in splitexp.split(line)[:-9]]
        doc_words = splitline[1::2]
        doc_counts = splitline[2::2]
        # print line
        for i in range(len(doc_words)):
            if doc_words[i] not in apperance_dict:
                apperance_dict[doc_words[i]] = doc_counts[i]
            else:
                apperance_dict[doc_words[i]] += doc_counts[i]
    for item in apperance_dict.keys():
        if apperance_dict[item] <= threshold:
            del apperance_dict[item]
    # 打开vocab，去掉常见词
    print len(apperance_dict)
    vocab = open(vocab_path).readlines()

    rearrange_vocab = [0 for i in range(len(vocab) + 1)]
    counter = 1
    for word, id in zip(vocab, range(len(vocab))):
        word = word.strip('\n')
        if id + 1 not in apperance_dict:
            output_vocab.write(word + '\n')
            rearrange_vocab[id + 1] = counter
            counter += 1
    return apperance_dict, rearrange_vocab


def collect_too_common_word(vocab_path, vect_path, output_vocab, threshold):
    if not os.path.exists(vect_path):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % vect_path
    splitexp = re.compile(r'[ :]')
    apperance_dict = {}
    for line in file(vect_path):
        line = line.strip()
        splitline = [int(i) for i in splitexp.split(line)]
        doc_words = splitline[1::2]
        doc_counts = splitline[2::2]
        print line
        for i in range(len(doc_words)):
            if doc_words[i] not in apperance_dict:
                apperance_dict[doc_words[i]] = doc_counts[i]
            else:
                apperance_dict[doc_words[i]] += doc_counts[i]
    for item in apperance_dict.keys():
        if apperance_dict[item] <= threshold:
            del apperance_dict[item]
    # 打开vocab，去掉常见词
    print len(apperance_dict)
    vocab = open(vocab_path).readlines()

    rearrange_vocab = [0 for i in range(len(vocab) + 1)]
    counter = 1
    for word, id in zip(vocab, range(len(vocab))):
        word = word.strip('\n')
        if id + 1 not in apperance_dict:
            output_vocab.write(word + '\n')
            rearrange_vocab[id + 1] = counter
            counter += 1
    return apperance_dict, rearrange_vocab


def collect_non_chinese(vocab_path, output_vocab):
    vocab = open(vocab_path).readlines()
    first_line = True
    non_chinese = {}
    rearrange_vocab = [0 for i in range(len(vocab))]
    counter = 1
    for word in vocab:
        if first_line is True:
            first_line = False
            continue
        split = word.strip('\n').rfind(':')
        str = word[0:split]
        id = int(word[split + 1:])
        if str_has_non_chinese(str):
            non_chinese[id] = str
        else:
            output_vocab.write(str + "\n")
            rearrange_vocab[id] = counter
            counter += 1
    return non_chinese, rearrange_vocab


if __name__ == "__main__":
    vocab_path = "E:\LabWork\hdp\data\\national_day\\allWord.dat"
    data_path = "E:\LabWork\hdp\data\\national_day\\wordVect.dat"
    # no_noise_vocab = open("E:\LabWork\hdp\data\\national_day\\allWord_no_noise.dat", "w+")
    # no_noise_file = open("E:\LabWork\hdp\data\\national_day\\wordVect_no_noise.dat", "w+")

    no_noise_vocab_path = "E:\LabWork\hdp\data\\national_day\\allWord_no_noise.dat"
    no_noise_input_path = "E:\LabWork\hdp\data\\national_day\\wordVect_no_noise.dat"

    # no_noise_no_common_file = open("E:\LabWork\hdp\data\\national_day\\wordVect_no_noise_no_common.dat", "w+")
    # no_noise_no_common_vocab = open("E:\LabWork\hdp\data\\national_day\\allWord_no_noise_no_common.dat", "w+")
    #
    no_noise_no_common_path = "E:\LabWork\hdp\data\\national_day\wordVect_no_noise_no_common.dat"
    no_noise_no_short_path = open("E:\LabWork\hdp\data\\national_day\wordVect_no_noise_no_common_no_short.dat", "w+")
    #
    # non_chinese, non_chinese_rearrange_vocab = collect_non_chinese(vocab_path, no_noise_vocab)
    # remove_noise_with_time_location(data_path, no_noise_file, non_chinese, non_chinese_rearrange_vocab)
    #
    # common, non_common_rearrange_vocab = collect_too_common_word_with_time_location(no_noise_vocab_path,
    #                                                                                 no_noise_input_path,
    #                                                                                 no_noise_no_common_vocab, 3000)
    # remove_too_common_word_with_time_location(no_noise_input_path, no_noise_no_common_file, common, non_common_rearrange_vocab)
    #
    remove_too_short_tweet_with_time_location(no_noise_no_common_path, no_noise_no_short_path, 3)
