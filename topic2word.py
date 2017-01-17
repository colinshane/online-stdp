import heapq
import time
import numpy as np
from operator import itemgetter
# vocab_path = "E:/LabWork/hdp/data/ap/vocab.txt"
# vocab_path = "E:\LabWork\hdp\\data\\sogou\\allWord.dat"
# vocab_path = "E:\LabWork\hdp\data\weibo_3M\AllWord.txt"
# vocab_path = "E:\LabWork\hdp\data\\weibo_3M\\allWord_no_noise_no_common.dat"
vocab_path = "E:\LabWork\hdp\data\\national_day\\allWord_no_noise_no_common.dat"
topic_path = "E:\LabWork\hdp\online-hdp\\None\\corpus-national_day_weibo_no_noise_no_common_T_300-kappa-0.8-tau-1-batchsize-100\\final.topics"
# topic_path = "E:\LabWork\hdp\online-hdp\None\corpus-sougou-kappa-0.5-tau-1-batchsize-10\\final.topics"
# topic_path = "E:\LabWork\lda\onlineldavb\lambda-final"
result_file = open("E:\LabWork\hdp\online-hdp\\None\\corpus-national_day_weibo_no_noise_no_common_T_300-kappa-0.8-tau-1-batchsize-100\\final.topics_words", "w+")
# result_file = file("E:\LabWork\hdp\online-hdp\None\corpus-sougou-kappa-0.5-tau-1-batchsize-10\\final.topics_words", "w")
# result_file = file("E:\LabWork\lda\onlineldavb\lambda-final.topics_words", "w")
ISOTIMEFORMAT='%Y-%m-%d %X'

def getTime():
    return time.strftime(ISOTIMEFORMAT, time.localtime(time.time()))

def convert_ap(vocab_path, topic_path, result_file):
    vocab_list = []
    topic_dict = {}
    for line in open(vocab_path):
        vocab_list.append(line.strip('\n'))
    i = 1
    for line in open(topic_path):
        print "%s processing topic %d" % (getTime(), i)
        splitexp = line.split(" ")
        splitexp = np.array([float(x) for x in splitexp])
        splitexp = splitexp / sum(splitexp)
        splitdict = dict(zip(range(len(splitexp)), splitexp))
        k_keys_sorted = heapq.nlargest(20, splitdict.items(), key=itemgetter(1))
        result_file.write("topic %d:\n" % i)
        for id_tuple in k_keys_sorted:
            result_file.write('%20s %d \t---\t  %.4f\n' % (vocab_list[id_tuple[0]], id_tuple[0],  id_tuple[1]))
        result_file.write("\n")
        i += 1
    result_file.flush()
    result_file.close()

def convert(vocab_path, topic_path, result_file):
    vocab_list = []
    topic_dict = {}
    flag = True
    for line in open(vocab_path):
        if flag:
            flag = False
            vocab_list = [0 for x in range(int(line) + 1)]
            continue
        split = line.strip('\n').rfind(':')
        vocab_list[int(line[split+1:])] = line[0:split]
    i = 1
    for line in open(topic_path):
        print "%s processing topic %d" % (getTime(), i)
        splitexp = line.split(" ")
        splitexp = np.array([float(x) for x in splitexp])
        splitexp = splitexp / sum(splitexp)
        splitdict = dict(zip(range(len(splitexp)), splitexp))
        k_keys_sorted = heapq.nlargest(20, splitdict.items(), key=itemgetter(1))
        result_file.write("topic %d:\n" % i)
        for id_tuple in k_keys_sorted:
            result_file.write('%20s \t---\t  %.6f\n' % (vocab_list[id_tuple[0]], id_tuple[1]))
        result_file.write("\n")
        i += 1
    result_file.flush()
    result_file.close()

def convert_no_noise(vocab_path, topic_path, result_file, vocab_length):
    vocab_list = [0]
    topic_dict = {}
    for line in open(vocab_path):
        vocab_list.append(line.strip('\n'))
    i = 1
    for line in open(topic_path):
        print "%s processing topic %d" % (getTime(), i)
        splitexp = line.split(" ")
        splitexp = np.array([float(x) for x in splitexp])
        splitexp = splitexp / sum(splitexp)
        splitdict = dict(zip(range(len(splitexp)), splitexp))
        k_keys_sorted = heapq.nlargest(20, splitdict.items(), key=itemgetter(1))
        result_file.write("topic %d:\n" % i)
        for id_tuple in k_keys_sorted:
            result_file.write('%20s %d \t---\t  %.4f\n' % (vocab_list[id_tuple[0]], id_tuple[0],  id_tuple[1]))
        result_file.write("\n")
        i += 1
    result_file.flush()
    result_file.close()

def convert_onlinelda(vocab_path, topic_path, result_file):
    vocab_list = []
    topic_dict = {}
    flag = True
    for line in open(vocab_path):
        if flag:
            flag = False
            vocab_list = [0 for x in range(int(line) + 1)]
            continue
        split = line.strip('\n').rfind(':')
        vocab_list[int(line[split+1:])] = line[0:split]

    testlambda = np.loadtxt(topic_path)

    for k in range(0, len(testlambda)):
        lambdak = list(testlambda[k, :])
        lambdak = lambdak / sum(lambdak)
        temp = zip(lambdak, range(0, len(lambdak)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        result_file.write('topic %d:' % (k))
        # feel free to change the "53" here to whatever fits your screen nicely.
        for i in range(0, 21):
            result_file.write('%20s  \t---\t  %.4f\n' % (vocab_list[temp[i][1]], temp[i][0]))
if __name__ == '__main__':
    convert_no_noise(vocab_path, topic_path, result_file, 999999)
    # convert_ap(vocab_path, topic_path, result_file)
    result_file.close()