import cPickle

vocab_path = "E:\LabWork\hdp\\test_data\\allWord.dat"
# topic_path = "E:/LabWork/hdp/online-hdp/None/corpus-nature/doc_count-35936.topics"
topic_path = "E:\LabWork\hdp\online-hdp\None\corpus-weibo_test-kappa-0.5-tau-1-batchsize-1\docu_count-1.topics"

# result_file = file("E:/LabWork/hdp/online-hdp/None/corpus-nature/doc_count-35936.topics_words", "w")
#result_file = file("E:\LabWork\hdp\online-hdp\None\corpus-weibo_test-kappa-0.5-tau-1-batchsize-1\docu_count-1.topics_words", "w")

test_data_path = "E:\LabWork\hdp\\online-hdp\\ap\\ap.txt"

# ohdp = cPickle.load(open("E:\LabWork\hdp\online-hdp\None\corpus-weibo-kappa-0.5-tau-1-batchsize-10000\init.model", "rb"))
# ohdp.save_topics(topic_path)
#
# topic2word.convert(vocab_path, topic_path, result_file)



# ohdp = cPickle.load(open("E:\LabWork\hdp\online-hdp\None\corpus-weibo_test-kappa-0.5-tau-1-batchsize-1\\final.model", "rb"))
hdp = cPickle.load(open("E:\LabWork\hdp\original\online-hdp\\test\corpus-sougou_seq-kappa-0.5-tau-1-batchsize-100\\final.model", "rb"))
#ohdp.hdp_to_lda()
(lda_alpha, lda_beta) = hdp.hdp_to_lda()
# hdp.save_topics("E:\LabWork\hdp\online-hdp\None\corpus-sougou-kappa-0.5-tau-1-batchsize-10\final.topics")



# predict on the fixed corpus
# test_data = read_data(test_data_path)
# test_score = 0.0
# test_score_split = 0.0
# predict_result_path = "E:\LabWork\hdp\online-hdp\None\corpus-nature\predict.gamma"
# predict_result_file = file(predict_result_path)
# gamma =
# for doc, i in zip(test_data.docs, range(0, len(test_data.docs))):
#     print "predicting doc %d" % i
#     (likelihood, gamma) = hdp.lda_e_step(doc, lda_alpha, lda_beta)
#     predict_result_file.write()
#     test_score += likelihood

# get all location





