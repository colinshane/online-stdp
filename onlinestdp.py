# coding=utf-8
"""
online hdp with lazy update
part of code is adapted from Matt's online lda code
"""
import numpy as np
import scipy.special as sp
import scipy.stats
import os, sys, math, time
import utils
from corpus import document, corpus
from itertools import izip
import random

ISOTIMEFORMAT = '%Y-%m-%d %X'


def getTime():
    return time.strftime(ISOTIMEFORMAT, time.localtime(time.time()))


meanchangethresh = 0.00001
random_seed = 999931111
np.random.seed(random_seed)
random.seed(random_seed)
mu0 = 0.3
rhot_bound = 0.0


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if len(alpha.shape) == 1:
        return sp.psi(alpha) - sp.psi(np.sum(alpha))
    elif len(alpha.shape) == 2:
        return sp.psi(alpha) - sp.psi(np.sum(alpha, 1))[:, np.newaxis]
    else:  # len(alpha.shape) == 3:
        return sp.psi(alpha) - sp.psi(np.sum(alpha, 2))[:, :, np.newaxis]

def expect_log_sticks(sticks):
    """
    For stick-breaking hdp, this returns the E[log(sticks)] 
    """
    if len(sticks.shape) == 2:
        dig_sum = sp.psi(np.sum(sticks, 0))
        ElogW = sp.psi(sticks[0]) - dig_sum
        Elog1_W = sp.psi(sticks[1]) - dig_sum

        n = len(sticks[0]) + 1 # len(sticks[0]) = T-1
        Elogsticks = np.zeros(n)
        Elogsticks[0:n - 1] = ElogW # 最后一位是0，因为q(beta_T')=1，E[log_q_beta_T']=0
        Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    elif len(sticks.shape) == 3:
        dig_sum = sp.psi(np.sum(sticks, 1))
        ElogW = sp.psi(sticks[:, 0, :]) - dig_sum
        Elog1_W = sp.psi(sticks[:, 1, :]) - dig_sum

        n = len(sticks[0, 0]) + 1
        Elogsticks = np.zeros((len(sticks), n))
        Elogsticks[:, 0:n - 1] = ElogW
        Elogsticks[:, 1:] = Elogsticks[:, 1:] + np.cumsum(Elog1_W, 1)


    return Elogsticks


def lda_e_step_half(doc, alpha, Elogbeta, split_ratio):
    n_train = int(doc.length * split_ratio)
    n_test = doc.length - n_train

    # split the document
    words_train = doc.words[:n_train]
    counts_train = doc.counts[:n_train]
    words_test = doc.words[n_train:]
    counts_test = doc.counts[n_train:]

    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(dirichlet_expectation(gamma))

    expElogbeta = np.exp(Elogbeta)
    expElogbeta_train = expElogbeta[:, words_train]
    phinorm = np.dot(expElogtheta, expElogbeta_train) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    max_iter = 100
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        gamma = alpha + expElogtheta * np.dot(counts / phinorm, expElogbeta_train.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, expElogbeta_train) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < meanchangethresh):
            break
    gamma = gamma / np.sum(gamma)
    counts = np.array(counts_test)
    expElogbeta_test = expElogbeta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, expElogbeta_test) + 1e-100))

    return (score, np.sum(counts), gamma)


def lda_e_step_split(doc, alpha, beta, max_iter=100):
    half_len = int(doc.length / 2) + 1
    idx_train = [2 * i for i in range(half_len) if 2 * i < doc.length]
    idx_test = [2 * i + 1 for i in range(half_len) if 2 * i + 1 < doc.length]

    # split the document
    words_train = [doc.words[i] for i in idx_train]
    counts_train = [doc.counts[i] for i in idx_train]
    words_test = [doc.words[i] for i in idx_test]
    counts_test = [doc.counts[i] for i in idx_test]

    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(dirichlet_expectation(gamma))
    betad = beta[:, words_train]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts / phinorm, betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < meanchangethresh):
            break

    gamma = gamma / np.sum(gamma)
    counts = np.array(counts_test)
    betad = beta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, betad) + 1e-100))

    return (score, np.sum(counts), gamma)


def lda_e_step(doc, alpha, beta, max_iter=100):
    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(dirichlet_expectation(gamma))
    betad = beta[:, doc.words]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(doc.counts)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts / phinorm, betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma - lastgamma))
        if (meanchange < meanchangethresh):
            break

    likelihood = np.sum(counts * np.log(phinorm))
    likelihood += np.sum((alpha - gamma) * Elogtheta)
    likelihood += np.sum(sp.gammaln(gamma) - sp.gammaln(alpha))
    likelihood += sp.gammaln(np.sum(alpha)) - sp.gammaln(np.sum(gamma))

    return (likelihood, gamma)


# online版本的M步需要的统计量
class suff_stats:
    def __init__(self, T, K, Wt, Dt):
        self.m_batchsize = Dt
        self.m_var_sticks_1st_ss = np.zeros(T)
        self.m_var_beta_ss = np.zeros([T, K, Wt])
        self.m_var_beta_noise_ss = np.zeros(Wt)
        self.m_var_sticks_2nd_ss = np.zeros((T, K))

        self.m_var_mu_time_ss_numerator = np.zeros((T, K))
        self.m_var_mu_time_ss_denominator = np.zeros((T, K))
        self.m_var_sigma_time_ss_numerator = np.zeros((T, K))
        self.m_var_sigma_time_ss_denominator = np.zeros((T, K))
        self.m_var_mu_location_ss_numerator = np.zeros([T, K, 2])
        self.m_var_mu_location_ss_denominator = np.zeros([T, K, 2])
        self.m_var_sigma_location_ss_numerator = np.zeros([T, K, 2, 2])
        self.m_var_sigma_location_ss_denominator = np.zeros([T, K, 2, 2])

    def set_zero(self):
        self.m_var_sticks_1st_ss.fill(0.0)
        self.m_var_sticks_2nd_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)
        self.m_var_beta_noise_ss.fill(0.0)
        self.m_var_mu_time_ss_numerator.fill(0.0)
        self.m_var_mu_time_ss_denominator.fill(0.0)
        self.m_var_sigma_time_ss_numerator.fill(0.0)
        self.m_var_sigma_time_ss_denominator.fill(0.0)
        self.m_var_mu_location_ss_numerator.fill(0.0)
        self.m_var_mu_location_ss_denominator.fil(0.0)
        self.m_var_sigma_location_ss_numerator.fill(0.0)
        self.m_var_sigma_location_ss_denominator.fill(0.0)

class online_stdp:
    ''' stdp model using stick breaking '''

    def __init__(self, T, K, D, W, eta, alpha, gamma, real_gamma, kappa, tau, scale=1.0, adding_noise=False):
        """
        this follows the convention of the HDP paper
        gamma: first level concentration
        alpha: second level concentration
        real_gamma: the prior paramater of the noise distribution
        eta: the topic Dirichlet
        T: Event level truncation level
        K: Element level truncation level
        W: size of vocab
        D: number of documents in the corpus
        kappa: learning rate
        tau: slow down parameter
        """

        self.m_W = W
        self.m_D = D
        self.m_T = T
        self.m_K = K
        self.m_alpha = alpha
        self.m_gamma = gamma
        self.m_real_gamma = real_gamma

        # 事件层的stick对应的两个参数
        self.m_var_sticks_1st = np.zeros((2, T - 1))  # 保存第一层stick的变分参数u,v的更新公式的sigma部分
        self.m_var_sticks_1st[0] = 1.0
        # self.m_var_sticks[1] = self.m_gamma
        # make a uniform at beginning
        self.m_var_sticks_1st[1] = range(T - 1, 0, -1)

        # 节点层的stick对应的两个参数
        self.m_var_sticks_2nd = np.zeros((T, 2, K - 1))
        self.m_var_sticks_2nd[:, 0, :] = 1.0
        self.m_var_sticks_2nd[:, 1, :] = range(K - 1, 0, -1)

        # 保存更新两层a,b,u,v的不算超参数的部分
        self.m_varphi_ss = np.zeros(T)
        self.m_varphi_phi_ss = np.zeros((T, K))

        # 这里使用一个二维的多项分布来近似，和笔记不同
        # self.m_Elogx = dirichlet_expectation(self.m_real_gamma)
        self.m_real_gamma = np.array([real_gamma, real_gamma])
        self.m_Elogx = sp.psi(self.m_real_gamma) - sp.psi(self.m_real_gamma)

        # T x K x W
        # self.m_lambda = np.random.gamma(1.0, 1.0, (T, W)) * D*100/(T*W)-eta
        self.m_lambda = np.random.gamma(1.0, 1.0, (T, K, W)) * D * 100 / (T * K * W) - eta
        self.m_lambda_noise = np.random.gamma(1.0, 1.0, W) * D * 100 / W - eta
        self.m_eta = eta
        # the topics Eq[log(P(w_jn=w|phi_e_k))] batch版本没有加上m_eta
        self.m_Elogbeta = dirichlet_expectation(self.m_eta + self.m_lambda)
        self.m_Elogbeta_noise = dirichlet_expectation(self.m_eta + self.m_lambda_noise)


        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_scale = scale
        self.m_updatect = 0
        self.m_status_up_to_date = True
        self.m_num_docs_parsed = 0

        # Timestamps and normalizers for lazy updates
        self.m_timestamp = np.zeros(self.m_W, dtype=int)  # 每个单词对应一个timestamp
        # 保存累加的log(1-rho)
        self.m_r = [0]
        # 保存每一步的log(1-rho)
        self.m_r_step = [0]
        self.m_lambda_sum = np.sum(self.m_lambda, axis=2)  # np.shape(m_lambda_sum) = (T, K)

    def time2timestamp(self, time):
        return int(time.mktime(time.strptime(time, "%Y-%m-%d %H:%M:%S")))

    def init_time_location_parameters(self, all_docs):
        # 均匀取T * K个点作为初值
        # self.m_mu_t = np.random.uniform(time2timestamp(time_range[0]), time2timestamp(time_range[1]), self.T * self.K).reshape(self.T, self.K)
        self.sample_docs = np.array(random.sample(all_docs.docs, self.m_T * self.m_K)).reshape(self.m_T, self.m_K)
        self.m_mu_t = np.array([[self.sample_docs[t][k].time for k in range(self.m_K)] for t in range(self.m_T)])
        self.m_sigma_t = np.ones((self.m_T, self.m_K))

        self.m_mu_l = np.array([[(self.sample_docs[t][k].latitude, self.sample_docs[t][k].longitude) for k in range(self.m_K)] for t in range(self.m_T)])
        self.m_sigma_l = np.array([[((1,0),(0,1)) for k in range(self.m_K)] for t in range(self.m_T)])

        # 由于q分布里面不包含时间的高斯分布，因此Elogtime = logtime
        # self.m_Elogtime = np.log(np.array([[scipy.stats.norm(self.m_mu_t[t][k], self.m_sigma_t[t][k] )]]))

    # unseen_ids表示docs中第一次取的doc, var_converage判断是否收敛
    def process_documents(self, batch_head, docs, var_converge, unseen_ids=[], update=True, opt_o=True):
        # Find the unique words in this mini-batch of documents...
        self.m_num_docs_parsed += len(docs)

        unique_words = dict()
        word_list = []

        for doc in docs:
            for w in doc.words:
                if w not in unique_words:
                    unique_words[w] = len(unique_words)  # 保存了这个单词对应的这一个minibatch的id
                    word_list.append(w)

        Wt = len(word_list)  # length of UNNIQUE words in these documents

        # ...and do the lazy updates on the necessary columns of lambda
        # m_r以timestamp为索引,保存叠加的log(1-rho),rw保存每个单词上一次更新的累加的log(1-rho)
        rw = np.array([self.m_r[t] for t in self.m_timestamp[word_list]])  # 取出单词表中单词的最后更新的轮数的m_r

        # ？？？这里没有考虑需要lazy_update的单词的eta
        # self.mr[-1]表示最新的累加的log(1-rho), 此处更新之前上一轮没有更新的单词的lambda
        self.m_lambda[:, :, word_list] *= np.exp(self.m_r[-1] - rw)

        self.m_Elogbeta[:, :, word_list] = sp.psi(self.m_eta + self.m_lambda[:, :, word_list]) - \
                                        sp.psi(self.m_W * self.m_eta + self.m_lambda_sum[:, :, np.newaxis])

        ss = suff_stats(self.m_T, self.m_K, Wt, len(docs))

        Elogsticks_1st = expect_log_sticks(self.m_var_sticks_1st)  # Eq[log_beta]
        Elogsticks_2nd = expect_log_sticks(self.m_var_sticks_2nd)  # Eq[log_pi]

        # run variational inference on some new docs
        score = 0.0
        count = 0
        unseen_score = 0.0
        unseen_count = 0
        for i, doc in enumerate(docs):
            if (i % 1000 == 0):
                print "%s [batch_head = %d] do E step of %d doc." % (getTime(), batch_head, i)
            doc_score = self.doc_e_step(i + batch_head, doc, ss, Elogsticks_1st, Elogsticks_2nd, word_list, unique_words,
                                        var_converge)
            count += doc.total
            score += doc_score
            if i in unseen_ids:
                unseen_score += doc_score
                unseen_count += doc.total

        if update:
            self.update_lambda(ss, word_list, opt_o)

        return (score, count, unseen_score, unseen_count)

    def optimal_ordering(self):
        """
        ordering the topics
        """
        # np.shape(self.m_lambda_sum) = (T, K)
        # 每个e中k的排序, np.shape(idek) = (T, K)
        idek = np.array([i for i in reversed(np.argsort(self.m_lambda_sum, axis=1))])
        # 每个e的排序 np.shape(ide) = (T,)
        ide = np.array([i for i in reversed(np.argsort(np.sum(self.m_lambda_sum, 1)))])
        # 然后将idek按照ide的顺序排列，这样每个T x K的矩阵就是先按照T排序然后按照K排序
        idek = idek[ide]

        self.m_varphi_ss = self.m_varphi_ss[ide]
        self.m_varphi_phi_ss = self.m_varphi_phi_ss[ide, :]
        self.m_varphi_phi_ss = np.array([self.m_varphi_phi_ss[i][idek[i]] for i in range(self.m_T)])

        self.m_lambda = self.m_lambda[ide, :, :]
        self.m_lambda = np.array([self.m_lambda[i][idek[i]][:] for i in range(self.m_T)])

        self.m_lambda_sum = self.m_lambda_sum[ide, :]
        self.m_lambda_sum = np.array([self.m_lambda_sum[i][idek[i]] for i in range(self.m_T)])

        self.m_Elogbeta = self.m_Elogbeta[ide, :, :]
        self.m_Elogbeta = np.array([self.m_Elogbeta[i][idek[i]][:] for i in range(self.m_T)])

        self.m_mu_t = self.m_mu_t[ide, :]
        self.m_mu_t = np.array([self.m_mu_t[i][idek[i]] for i in range(self.m_T)])

        self.m_sigma_t = self.m_sigma_t[ide, :]
        self.m_sigma_t = np.array([self.m_sigma_t[i][idek[i]] for i in range(self.m_T)])

        self.m_mu_l = self.m_mu_l[ide, :, :]
        self.m_mu_l = np.array([self.m_mu_l[i][idek[i]][:] for i in range(self.m_T)])

        self.m_sigma_l = self.m_sigma_l[ide, :, :, :]
        self.m_sigma_l = np.array([self.m_sigma_l[i][idek[i]][:][:] for i in range(self.m_T)])


    def doc_e_step(self, batch_count, doc, ss, Elogsticks_1st, Elogsticks_2nd, word_list, unique_words, var_converge,
                       max_iter=500):
        """
        e step for a single doc
        """
        batchids = [unique_words[id] for id in doc.words]  # 生成文档对应的这个batch的id,不是全局的id(doc.words保存的是全局id)
        Elogbeta_doc = self.m_Elogbeta[:, :, doc.words]
        Elogbeta_doc_noise = self.m_Elogbeta_noise[doc.words]
        # Elogtime_doc.shape = (T, K)
        Elogtime_doc = np.array([[scipy.stats.norm.logpdf(doc.time, self.m_mu_t[t][k], self.m_sigma_t[t][k])\
                                  for k in range(self.m_K)] for t in range(self.m_T)])
        # Eloglocation_doc.shape = (T, K)
        Eloglocation_doc = np.array([[scipy.stats.multivariate_normal.logpdf((doc.latitude, doc.longitude),\
                                                                             self.m_mu_l[t][k], self.m_sigma_l[t][k])\
                                      for k in range(self.m_K)] for t in range(self.m_T)])

        # 初始化x_hat
        # np.shape(x_hat) = (doc.length)
        x_hat = np.ones(doc.length) / 2 # 每个词都初始化为0.5
        x_hat_bar = x_hat

        # 表示微博属于不同节点的概率，初始化为均匀分布，微博属于不同节点的概率都相同
        # np.shape(phi) = (T, K)
        phi = np.ones((self.m_T, self.m_K)) * 1.0 / self.m_K
        # 表示微博属于不同事件的概率
        # np.shape(Elogbeta_doc * doc.counts) = (T, K, N),
        # np.shape(x_hat) = (N,) = (N x 1)
        # np.shape(var_phi) = (T,) = (T x 1)
        var_phi = np.sum(np.dot((Elogbeta_doc * doc.counts), x_hat) * phi, 1)

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0
        eps = 1e-100

        iter = 0
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            if iter < 3:
                var_phi = np.sum(np.dot((Elogbeta_doc * doc.counts), x_hat) * phi, 1)
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.sum(np.dot((Elogbeta_doc * doc.counts), x_hat) * phi, 1) + Elogsticks_1st \
                          + np.sum(phi * Elogtime_doc, 1) + np.sum(phi * Eloglocation_doc, 1)
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)

            if iter < 3:
                phi = np.dot((Elogbeta_doc * doc.counts), x_hat)
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot((Elogbeta_doc * doc.counts), x_hat) + Elogsticks_2nd + Eloglocation_doc + Eloglocation_doc
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)

            # 更新x_hat 这里使用一个二维的多项分布来近似，和笔记不同
            # 先转置三维数组，(T, K, N)转换为(N, T, K)然后和phi逐项相乘，然后K那一维相加,最后乘以var_phi
            x_hat = self.m_Elogx[0] + np.dot(np.sum(np.transpose(Elogbeta_doc * doc.counts, (2, 0, 1)) * phi, 2), var_phi)
            x_hat_bar = self.m_Elogx[1] + Elogbeta_doc_noise
            # 然后合并两个矩阵，用log_normalize归一化，得到 N x 2的矩阵
            (log_x_hat, log_norm) = utils.log_normalize(np.column_stack((x_hat, x_hat_bar)))
            x_hat_final = np.exp(log_x_hat)
            # 再分开两列，后面要用
            log_x_hat_bar = log_x_hat[:, 1]
            log_x_hat = log_x_hat[:, 0]
            x_hat = x_hat_final[:, 0]
            x_hat_bar = x_hat_final[:, 1]

            likelihood = 0.0
            # compute likelihood
            # 取文档内的参数的似然相加
            # 展开式的第五项和第七项相加, # np.shape(Elogsticks_1st) = (T,) np.shape(log_var_phi) = (T,) = np.shape(var_phi)
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)
            # 展开式的第六项和第八项相加  # np.shape(Elogsticks_2nd) = (T, K) np.shape(log_phi) = (T, K) = np.shape(phi)
            likelihood += np.sum(np.sum((Elogsticks_2nd - log_phi) * phi, 1) * var_phi)
            # 展开式的第四项和第九项相加  # np.shape(self.m_Elogx) = (2,) np.shape(log_x_hat) = np.shape(x_hat) = (N,)
            likelihood += np.sum(np.sum(self.m_Elogx[0] - log_x_hat) * x_hat) + np.sum(np.sum(self.m_Elogx[1] - log_x_hat_bar) * x_hat_bar)
            # 展开式的第一项,分为两部分，一部分是噪声生成的词项，一部分是非噪声
            #  np.shape(Elogbeta_doc) = (T, K, N) np.shape(var_phi) = (T,) np.shape(Elogbeta_doc_noise) = (N,)
            likelihood += np.sum(np.sum(np.dot((Elogbeta_doc * doc.counts), x_hat) * phi, 1) * var_phi) + \
                np.dot((Elogbeta_doc_noise * doc.counts), x_hat_bar)
            # 展开式的第二项
            likelihood += np.sum(np.sum(Elogtime_doc * phi, 1) * var_phi)
            # 展开式的第三项
            likelihood += np.sum(np.sum(Eloglocation_doc * phi, 1) * var_phi)

            converge = (likelihood - old_likelihood) / abs(old_likelihood)

            if converge < -0.000001:
                print "%s [batch_count = %d iter = %d] warning, likelihood is decreasing! old_likelihood = %f new_likelihood = %f" % (getTime(), batch_count, iter, old_likelihood, likelihood)

            old_likelihood = likelihood

            iter += 1
        # print "%s [batch_count = %d iter = %d]  new_likelihood = %f" % (getTime(), batch_count, iter, likelihood)

        # update the suff_stat ss
        # this time it only contains information from one doc
        st = phi * var_phi[:, np.newaxis]
        ss.m_var_sticks_1st_ss += var_phi
        ss.m_var_sticks_2nd_ss += st
        #
        ss.m_var_beta_ss[:, :, batchids] += np.ones([self.m_T, self.m_K, doc.length]) * x_hat * doc.counts * ((phi * var_phi[:, np.newaxis])[:, :, np.newaxis])
        ss.m_var_beta_noise_ss[batchids] += x_hat_bar * doc.counts
        #
        ss.m_var_mu_time_ss_numerator += st * doc.time
        ss.m_var_mu_time_ss_denominator += st
        ss.m_var_sigma_time_ss_numerator += st * pow(doc.time - self.m_mu_t, 2)
        ss.m_var_sigma_time_ss_denominator += st
        # 矩阵逐项相乘 是后对齐
        ss.m_var_mu_location_ss_numerator += np.ones([self.m_T, self.m_K, 2]) * np.array([doc.latitude, doc.longitude]) * ((phi * var_phi[:, np.newaxis])[:, :, np.newaxis])
        ss.m_var_mu_location_ss_denominator += st[:, :, np.newaxis]
        ss.m_var_sigma_location_ss_numerator += np.array([[ np.array([(pow(doc.latitude - self.m_mu_l[t][k][0], 2), (doc.latitude - self.m_mu_l[t][k][0]) * (doc.longitude - self.m_mu_l[t][k][1])),\
                                                                      ((doc.latitude - self.m_mu_l[t][k][0]) * (doc.longitude - self.m_mu_l[t][k][1]), pow(doc.longitude - self.m_mu_l[t][k][1], 2))]) \
                                                            * phi[t][k] * var_phi[t] for k in range(self.m_K)] for t in range(self.m_T)])
        ss.m_var_sigma_location_ss_denominator += np.array([[ np.ones((2, 2))* phi[t][k] * var_phi[t] \
                                                              for k in range(self.m_K)] for t in range(self.m_T)])

        return (likelihood)

    def update_lambda(self, sstats, word_list, opt_o):
        self.m_status_up_to_date = False
        if len(word_list) == self.m_W:  # 如果word_list里面包含所有单词
          self.m_status_up_to_date = True
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = self.m_scale * pow(self.m_tau + self.m_updatect, -self.m_kappa)
        if rhot < rhot_bound:
            rhot = rhot_bound
        self.m_rhot = rhot

        # Update appropriate columns of lambda based on documents.
        self.m_lambda[:, :, word_list] = self.m_lambda[:, :, word_list] * (1 - rhot) + \
            rhot * self.m_D * sstats.m_var_beta_ss / sstats.m_batchsize
        self.m_lambda_sum = (1 - rhot) * self.m_lambda_sum + \
            rhot * self.m_D * np.sum(sstats.m_var_beta_ss, axis=2) / sstats.m_batchsize

        self.m_updatect += 1
        self.m_timestamp[word_list] = self.m_updatect  # 保存对应的lambda[:, word_list]更新的时间戳
        self.m_r.append(self.m_r[-1] + np.log(1 - rhot))  # 每一轮保存一个累加的log(1-rhot)

        coefficient = self.m_D / sstats.m_batchsize
        # 更新两层stick的不算超参数的部分
        self.m_varphi_ss = (1.0 - rhot) * self.m_varphi_ss + rhot * sstats.m_var_sticks_1st_ss * \
            self.m_D / sstats.m_batchsize
        self.m_varphi_phi_ss = (1.0 - rhot) * self.m_varphi_phi_ss + rhot * sstats.m_var_sticks_2nd_ss * \
            self.m_D / sstats.m_batchsize

        # 更新时间/空间部分
        self.m_mu_t = (1.0 - rhot) * self.m_mu_t + rhot * (sstats.m_var_mu_time_ss_numerator / sstats.m_var_mu_time_ss_denominator) * \
                                                   coefficient
        self.m_sigma_t = (1.0 - rhot) * self.m_sigma_t + rhot * (sstats.m_var_sigma_time_ss_numerator / sstats.m_var_sigma_time_ss_denominator) * \
                                                         coefficient
        self.m_mu_l = (1.0 - rhot) * self.m_mu_l + rhot * (sstats.m_var_mu_location_ss_numerator / sstats.m_var_mu_location_ss_denominator) * \
                                                   coefficient
        self.m_sigma_l = (1.0 - rhot) * self.m_sigma_l + rhot * (sstats.m_var_sigma_location_ss_numerator / sstats.m_var_sigma_location_ss_denominator) * \
                                                         coefficient

        if opt_o:
            self.optimal_ordering()
        # 这里排序之后紧接着m_var_sticks排序

        ## update top level sticks
        # 加两个超参数为什么要放在外面？
        self.m_var_sticks_1st[0] = self.m_varphi_ss[:self.m_T-1] + 1.0
        var_phi_sum = np.flipud(self.m_varphi_ss[1:]) # 根据公式，第一个不算，这样还剩T-1个
        self.m_var_sticks_1st[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma

        self.m_var_sticks_2nd[:, 0, :] = self.m_varphi_phi_ss[:, :self.m_K-1] + 1.0
        var_phi_phi_sum = np.fliplr(self.m_varphi_phi_ss[:, 1:])
        self.m_var_sticks_2nd[:, 1, :] = np.fliplr(np.cumsum(var_phi_phi_sum, axis=1)) + self.m_alpha

    def update_expectations(self):
        """
        Since we're doing lazy updates on lambda, at any given moment
        the current state of lambda may not be accurate. This function
        updates all of the elements of lambda and Elogbeta so that if (for
        example) we want to print out the topics we've learned we'll get the
        correct behavior.
        """
        for w in range(self.m_W):
            self.m_lambda[:, :, w] *= np.exp(self.m_r[-1] -
                                          self.m_r[self.m_timestamp[w]])

        self.m_Elogbeta = sp.psi(self.m_eta + self.m_lambda) - \
            sp.psi(self.m_W * self.m_eta + self.m_lambda_sum[:, :, np.newaxis])
        self.m_timestamp[:] = self.m_updatect
        self.m_status_up_to_date = True

    def save_topics(self, filename):
        if not self.m_status_up_to_date:
            self.update_expectations()
        f = file(filename + ".time", "w")
        for (mus, sigmas) in zip(self.m_mu_t, self.m_sigma_t):
            for (mu, sigma) in zip(mus, sigmas):
                line = str(mu) + ' ' + str(sigma)
                f.write(line + '\n')
        f.close()
        f = file(filename + ".location", "w")
        for (mus, sigmas) in zip(self.m_mu_l, self.m_sigma_l):
            for (mu, sigma) in zip(mus, sigmas):
                line = ' '.join([str(x) for x in mu])
                line += ' ' + ' '.join([str(x) for x in sigma.flatten])
        f.close()
        f = file(filename + ".lambda", "w")
        betas = self.m_lambda + self.m_eta
        for betax in betas:
            for beta in betax:
                line = ' '.join([str(x) for x in beta])
                f.write(line + '\n')
        f.close()
