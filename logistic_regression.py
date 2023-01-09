# CS542 Fall 2021 Programming Assignment 2
# Logistic Regression Classifier

import os
from sys import intern
import numpy as np
from collections import defaultdict
from math import ceil, log
from random import Random
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import sort

'''
Computes the logistic function.
'''
def sigma(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():

    def __init__(self, n_features=4):
        self.class_dict = {'false': 0, 'true': 1}
        self.n_features = n_features
        self.theta = np.zeros(n_features + 1) # weights (and bias)

    def load_data(self, data_set):
        filenames = []
        false_files = []
        true_files = []
        
        classes = dict()
        documents = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    if name != '.DS_Store' and 'train' in root:
                        filenames.append(name)
                        if 'false' in root:
                            false_files.append(name)
                            for item in false_files:
                                classes[item] = 'false'
                                if 'false' in root:
                                    with open(os.path.join(root,item)) as f:
                                        content = f.read()
                                        words = content.split()
                                        word_list = []
                                        for word in words:
                                            word_list.append(word)
                                        documents[item] = word_list
                        elif 'true' in root:
                            true_files.append(name)
                            for item in true_files:
                                classes[item] = 'true'
                                if 'true' in root:
                                    with open(os.path.join(root,item)) as f:
                                        content = f. read()
                                        words = content.split()
                                        word_list = []
                                        for word in words:
                                            word_list.append(word)
                                        documents[item] = word_list       
                    elif name != '.DS_Store' and 'test' in root:
                        filenames.append(name)

                        if 'false' in root:
                            false_files.append(name)
                            
                            for item in false_files:
                                classes[item] = 'false'
                                if 'flase' in root:
                                    with open(os.path.join(root,item)) as f:
                                        content = f.read()
                                        words = content.split()
                                        word_list = []
                                        for word in words:
                                            word_list.append(word)
                                        documents[item] = word_list
                        elif 'true' in root:
                            true_files.append(name)
                            
                            for item in true_files:
                                classes[item] = 'true'
                                if 'true' in root:
                                    with open(os.path.join(root,item)) as f:
                                        content = f. read()
                                        words = content.split()
                                        word_list = []
                                        for word in words:
                                            word_list.append(word)
                                        documents[item] = word_list     
                    elif name != '.DS_Store' and 'dev' in root:
                        filenames.append(name)
                        if 'false' in root:
                            false_files.append(name)
                            for item in false_files:
                                classes[item] = 'false'
                                if 'false' in root:
                                    with open(os.path.join(root,item)) as f:
                                        content = f.read()
                                        words = content.split()
                                        word_list = []
                                        for word in words:
                                            word_list.append(word)
                                        documents[item] = word_list
                        elif 'true' in root:
                            true_files.append(name)
                            for item in true_files:
                                classes[item] = 'true'
                                if 'true' in root:
                                    with open(os.path.join(root,item)) as f:
                                        content = f. read()
                                        words = content.split()
                                        word_list = []
                                        for word in words:
                                            word_list.append(word)
                                        documents[item] = word_list
        
        return filenames, classes, documents

    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        lcount_true = 0
        lcount_false = 0
        binary_code = 0
        total_words = 0
        feature_true = ['COVID-19','prosecute','video','Coronavirus','Geng','Shuang','estimated']
        feature_false = ['COVID-19','no','support','killing','false','prove','denied']
        for item in document:
            for i in feature_false:
                if i == item:
                    lcount_false = lcount_false + 1
            for k in feature_true:
                if k == item:
                    lcount_true = lcount_true + 1
            total_words = total_words + 1
        
        if 'COVID-19' in document:
            binary_code = 1
        else:
            binary_code = 0
        vector[0] = lcount_true
        vector[1] = lcount_false
        if lcount_false > lcount_true:
            vector[2] = 0
        else:
            vector[2] = 1
        #vector[3] = 2
        vector[3] = (lcount_true+lcount_false)/100
        vector[-1] = 1
        
        return vector

    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        f_list = []
        c_list = []
        loss_history= []
        desired_order = sorted(documents.keys())
        #resort the document
        documents = {k:documents[k] for k in desired_order}
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
                for kv in documents.items():
                    for n in minibatch:
                        if n == kv [0]:
                            value = self.featurize(kv[1])
                            f_list.append(value)
                            
                            
                for m in sorted(classes):
                    for k in minibatch:
                        if k == m:
                            c_list.append(self.class_dict.get(classes.get(m)))
                            
                
                feature_matrix = np.array(f_list)
                class_matrix = np.array(c_list).reshape(-1)
                feature_matrix= feature_matrix[i * batch_size: (i + 1) * batch_size]
                class_matrix = class_matrix[i * batch_size: (i + 1) * batch_size]
                y = sigma(feature_matrix @ self.theta)
                loss = -(class_matrix * np.log(y)+(1-class_matrix)*np.log(1-y))
                
                ave_gradient = (feature_matrix.T@(y-class_matrix))/feature_matrix.shape[0]
                self.theta = self.theta - eta * ave_gradient
                
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)
            loss_history.append(loss)
            print(loss_history)
            return loss_history



    def test(self, dev_set):
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        filenames = sorted(filenames)
        feature_list = []
        count = 0
        for name in filenames:
            results[name]['correct'] = self.class_dict.get(classes.get(name))
            
            for kv in documents.items():
                if name == kv[0]:
                    value = self.featurize(kv[1])
                    feature_list.append(value)
            feature_matrix = np.array(feature_list)
            feature_matrix = feature_matrix[count]
            
            y = feature_matrix @ self.theta
            if y > 0.5:
                predicted = 1
            else:
                predicted = 0
            results[name]['predicted'] = predicted
            count += 1
            
            
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
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
        len_results = len(results)
        acc = count / len_results
        ## for positive
        if(tp+fp == 0):
            precision_p = 0
        else:
            precision_p = tp / (tp + fp)
        
        if(fn+tn == 0):
            precision_n = 0
        else:
            precision_n = tn/(fn+tn)
        
        if(tp+fn == 0):
            recall_p = 0
        else:
            recall_p = tp / (tp + fn)
        
        if(tn+fp == 0):
            recall_n = 0
        else:
            recall_n = tn/(tn+fp)
        
        if (precision_p + recall_p == 0):
            F1_p = 0
        else:
            F1_p = 2 * precision_p * recall_p / (precision_p + recall_p)
        if(precision_n + recall_n == 0):
            F1_n = 0
        else:
            F1_n = 2 * precision_n * recall_n / (precision_n + recall_n)
        confusion_matrix[0, 0] = tp
        confusion_matrix[0, 1] = fn
        confusion_matrix[1, 0] = fp
        confusion_matrix[1, 1] = tn
        print('accuracy:',acc)
        print('precision for positive:',precision_p)
        print('precision for negative:',precision_n)
        print('recall for positive',recall_p)
        print('recall for negative',recall_n)
        print('F1 for positive:',F1_p)
        print('F1 for negative',F1_n)
        print('confusion_matrix: [[true positive, false negative]]')
        print('confusion_matrix: [[false positive, true negative]]')
        print(confusion_matrix)


if __name__ == '__main__':
    lr = LogisticRegression(n_features=4)
    # make sure these point to the right directories
    history = lr.train('fake_news/train', batch_size=2, n_epochs=10, eta=0.25)
    # plt.plot(np.arange(len(history)), np.array(history))
    # plt.show()
    #lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    results = lr.test('fake_news/dev')
    #results = lr.test('movie_reviews_small/test')
    lr.evaluate(results)
