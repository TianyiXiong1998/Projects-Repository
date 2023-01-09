# CS114B Spring 2021 Programming Assignment 1
# Naive Bayes Classifier and Evaluation

import os
import numpy as np
from collections import defaultdict
import collections


class NaiveBayes():

    def __init__(self):
        # be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
        #self.class_dict = {'action': 0, 'comedy': 1}
        self.feature_dict = {}
        self.prior = None
        self.logprior = None
        self.likelihood = None
        self.loglikelihood = None
        self.likelihood_all = None
        self.loglikelihood_all = None
        self.vocabulory = None
        self.logprior = self.prior
        self.p_frequency_dic = {}
        self.n_frequency_dic = {}
        self.a_frequency_dic = {}
        self.c_frequency_dic = {}

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''

    def train(self, train_set):

        n_documents = 0
        n_neg = 0  # neg中文件数
        n_pos = 0  # pos中文件数
        n_action = 0
        n_comedy = 0
        n_frequency_dic = {}
        p_frequency_dic = {}
        a_frequency_dic = {}
        c_frequency_dic = {}
        each_neg_word = []  # the list of every word of neg file
        each_pos_word = []
        each_act_word = []
        each_come_word = []
        neg_word = []
        pos_word = []
        act_word = []
        come_word = []
        word = []  # list of words in V
        logprior = None
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    if name != '.DS_Store' and 'train' in root:
                        if name != '.DS_Store' and 'false' in root:
                            for i in range(len(files)):
                                n_neg = n_neg + 1
                            for file in files:
                                with open(os.path.join(root, file)) as f:
                                    content = f.read()
                                    words = content.rsplit()
                                    each_neg_word.append(words)
                                    neg_word.extend(words)

                        if name != '.DS_Store' and'true' in root:
                            for j in range(len(files)):
                                n_pos = n_pos + 1
                            for file in files:
                                with open(os.path.join(root, file)) as f:

                                    content = f.read()
                                    words = content.rsplit()
                                    each_pos_word.append(words)
                                    pos_word.extend(words)

                        for name in files:
                            with open(os.path.join(root, name), encoding='unicode_escape') as f:
                                # collect class counts and feature counts
                                if root != 'fake_news/train' and name != '.DS_Store':
                                    n_documents = n_documents + 1
                                    content = f.read()
                                    words = content.rsplit(' ')
                                    word.extend(words)
        for item in each_neg_word:
            for i in item:
                if i in n_frequency_dic:
                    n_frequency_dic[i] = n_frequency_dic[i] + 1
                else:
                    n_frequency_dic[i] = 1

        for item in each_pos_word:
            for i in item:
                if i in p_frequency_dic:
                    p_frequency_dic[i] = p_frequency_dic[i] + 1
                else:
                    p_frequency_dic[i] = 1

        self.p_frequency_dic = p_frequency_dic
        self.n_frequency_dic = n_frequency_dic
        self.a_frequency_dic = a_frequency_dic
        self.c_frequency_dic = c_frequency_dic

        word = list(set(word))
        n_neg_word = len(neg_word)  # neg words
        n_pos_word = len(pos_word)  # positive words
        n_act_word = len(act_word)
        n_come_word = len(come_word)

        if n_pos and n_neg:
            pos1 = n_pos / n_documents
            neg1 = n_neg / n_documents
        elif n_action and n_comedy:
            act1 = n_action / n_documents
            come1 = n_comedy/ n_documents
        if n_pos_word and n_neg_word:
            self.prior = np.array([pos1,neg1],dtype=float)
            logprior = np.log(self.prior)
        elif n_act_word and n_come_word:
            self.prior = np.array([act1, come1],dtype=float)
            logprior = np.log(self.prior)
        self.logprior = logprior

        n_word = len(word)  # 总词数（不重复）

        self.vocabulory = word
        sum_neg = n_neg_word + n_word
        sum_pos = n_pos_word + n_word
        sum_act = n_act_word + n_word
        sum_com = n_come_word + n_word
        if p_frequency_dic:
            self.likelihood_all = np.zeros((2, n_word), dtype=float)
        elif a_frequency_dic:
            self.likelihood_all = np.zeros((2, n_word),dtype=float)
        if p_frequency_dic and n_frequency_dic:
            for (i, c) in enumerate(word):
                if c in p_frequency_dic.keys():
                    self.likelihood_all[0, i] = (p_frequency_dic.get(c) + 1) / sum_pos
                else:
                    self.likelihood_all[0, i] = 1 / sum_pos

                if c in n_frequency_dic.keys():
                    self.likelihood_all[1, i] = (n_frequency_dic.get(c) + 1) / sum_neg
                else:
                    self.likelihood_all[1, i] = 1 / sum_neg

        elif a_frequency_dic and c_frequency_dic:

            for (i, c) in enumerate(word):
                if c in a_frequency_dic.keys():
                    self.likelihood_all[0, i] = (a_frequency_dic.get(c) + 1) / sum_act
                else:
                    self.likelihood_all[0, i] = 1 / sum_act

                if c in c_frequency_dic.keys():
                    self.likelihood_all[1, i] = (c_frequency_dic.get(c) + 1) / sum_com
                else:
                    self.likelihood_all[1, i] = 1 / sum_com

        self.loglikelihood_all = np.log(self.likelihood_all)

        self.feature_dict = self.select_features(train_set)
        if p_frequency_dic:
            self.likelihood = np.zeros((2, len(self.feature_dict)), dtype=float)

        elif a_frequency_dic:
            self.likelihood = np.zeros((2, len(self.feature_dict)),dtype=float)
        if p_frequency_dic and n_frequency_dic:
            for (i, c) in enumerate(word):
                # 要是没有feature select就没有 后面的判断
                if c in p_frequency_dic.keys() and c in self.feature_dict.keys():
                    self.likelihood[0, self.feature_dict.get(c)] = (p_frequency_dic.get(c) + 1) / sum_pos

                elif c in self.feature_dict.keys():
                    self.likelihood[0, self.feature_dict.get(c)] = 1 / sum_pos

                if c in n_frequency_dic.keys() and c in self.feature_dict.keys():
                    self.likelihood[1, self.feature_dict.get(c)] = (n_frequency_dic.get(c) + 1) / sum_neg

                elif c in self.feature_dict.keys():
                    self.likelihood[1, self.feature_dict.get(c)] = 1 / sum_neg


        elif a_frequency_dic and c_frequency_dic:
            for (i, c) in enumerate(word):
                if c in a_frequency_dic.keys() and c in self.feature_dict.keys():
                    self.likelihood[0, self.feature_dict.get(c)] = (a_frequency_dic.get(c) + 1) / sum_act

                elif c in self.feature_dict.keys():
                    self.likelihood[0, self.feature_dict.get(c)] = 1 / sum_act

                if c in c_frequency_dic.keys() and c in self.feature_dict.keys():
                    self.likelihood[1, self.feature_dict.get(c)] = (c_frequency_dic.get(c) + 1) / sum_com

                elif c in self.feature_dict.keys():
                    self.likelihood[1, self.feature_dict.get(c)] = 1 / sum_com
        loglikelihood = np.log(self.likelihood)
        self.loglikelihood = loglikelihood
        print(self.logprior)
        print(self.loglikelihood)


    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''

    def test(self, dev_set):

        results = defaultdict(dict)

        logprior_p = self.logprior[0]
        logprior_n = self.logprior[1]

        logprior_a = self.logprior[0]
        logprior_c = self.logprior[1]
        neg = 0
        pos = 0
        neg_file = []
        pos_file = []

        act_file = []
        com_file = []
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):

            for name in files:
                with open(os.path.join(root, name)) as f:
                    # create feature vectors for each document
                    if 'false' in root:
                        neg_file.append(name)
                    elif 'true' in root:
                        pos_file.append(name)
                # get most likely class
            for item in neg_file:
                results[item]['correct'] = self.class_dict.get('false')
                if 'false' in root :
                    with open(os.path.join(root, item)) as f:

                        content = f.read()
                        words = content.split()

                        for word in words:
                            if word in self.vocabulory and word in self.feature_dict:
                                #ind = self.vocabulory.index(word)
                                ind = self.feature_dict.get(word)
                                neg = logprior_n + self.loglikelihood[1][ind]
                                pos = logprior_p + self.loglikelihood[0][ind]
                                pred_matri = np.array([neg, pos],dtype=float)

                                prediction = np.argmax(pred_matri)
                                results[item]['predicted'] = prediction


            for items in pos_file:
                results[items]['correct'] = self.class_dict.get('true')
                if 'true' in root:
                    with open(os.path.join(root, items)) as f:
                        content = f.read()
                        words = content.split()

                        for word in words:
                            if word in self.vocabulory and word in self.feature_dict:
                                #ind = self.vocabulory.index(word)
                                ind = self.feature_dict.get(word)
                                neg = logprior_n + self.loglikelihood[1][ind]
                                pos = logprior_p + self.loglikelihood[0][ind]

                                pred_matri = np.array([neg, pos],dtype=float)

                                prediction = np.argmax(pred_matri)
                                results[items]['predicted'] = prediction

        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''

    def evaluate(self, results):
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))
        count = 0
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        for i in results:
            if results.get(i).get('correct') == results.get(i).get('predicted'):
                count = count + 1
            if results.get(i).get('correct') == 0 and results.get(i).get('predicted') == 1:
                fp = fp + 1
            if results.get(i).get('correct') == 1 and results.get(i).get('predicted') == 1:
                tp = tp + 1
            if results.get(i).get('correct') == 1 and results.get(i).get('predicted') == 0:
                fn = fn + 1
            if results.get(i).get('correct') == 0 and results.get(i).get('predicted') == 0:
                tn = tn + 1
        print(fp,tp,fn,tn)
        len_results = len(results)
        acc = count / len_results
        
        if(tp+fp == 0):
            precision = 0
        else:
            precision = tp / (tp + fp)
        if(tp+fn == 0):
            recall = 0
        else:
            recall = tp / (tp + fn)
        if (precision + recall == 0):
            F1 = 0
        else:
            F1 = 2 * precision * recall / (precision + recall)
        confusion_matrix[0, 0] = tp
        confusion_matrix[0, 1] = fn
        confusion_matrix[1, 0] = fp
        confusion_matrix[1, 1] = tn
        print(acc)
        print(precision)
        print(recall)
        print(F1)
        print(confusion_matrix)

    '''
    Performs feature selection.
    Returns a dictionary of features.
    '''

    def select_features(self, train_set):
        # almost any method of feature selection is fine here
        class_first_word = []
        loglikelihood_all = self.loglikelihood_all

        p_words_sorted = sorted(self.p_frequency_dic.items(),key = lambda kv:(kv[1],kv[0]))
        n_words_sorted = sorted(self.n_frequency_dic.items(),key = lambda kv:(kv[1],kv[0]))
        a_words_sorted = sorted(self.a_frequency_dic.items(),key = lambda kv:(kv[1],kv[0]))
        c_words_sorted = sorted(self.c_frequency_dic.items(),key = lambda kv:(kv[1],kv[0]))
        
        #select mannually

        return {'wonderful': 0, 'bad': 1, 'great': 2, 'doubt': 3}


if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('fake_news/train')

    #nb.train('movie_reviews_small/train')
    results = nb.test('fake_news/dev')
    #results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)
