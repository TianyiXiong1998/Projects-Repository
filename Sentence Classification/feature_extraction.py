import numpy as np


class BagOfWord:
    def __init__(self, do_lower_case=False):
        self.vocab = {}
        self.do_lower_case = do_lower_case

    def fit(self, sent_list):
        # sent_list 类型为 List
        for sent in sent_list:
            if self.do_lower_case:
                sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)

    def transform(self, sent_list):
        vocab_size = len(self.vocab)
        bag_of_word_feature = np.zeros((len(sent_list), vocab_size))
        for idx, sent in enumerate(sent_list):
            if self.do_lower_case:
                sent = sent.lower()
            words = sent.strip().split(" ")
            for word in words:
                bag_of_word_feature[idx][self.vocab[word]] += 1
        return bag_of_word_feature

    def fit_transform(self, sent_list):
        self.fit(sent_list)
        return self.transform(sent_list)


class NGram:
    def __init__(self, ngram, do_lower_case=False):
        self.ngram = ngram
        self.feature_map = {}
        self.do_lower_case = do_lower_case

    def fit(self, sentList):
        for gram in self.ngram:
            for sent in sentList:
                if self.do_lower_case:
                    sent = sent.lower()
                sent = sent.split(" ")
                for i in range(len(sent) - gram + 1):
                    feature = "_".join(sent[i:i + gram])
                    if feature not in self.feature_map:
                        self.feature_map[feature] = len(self.feature_map) #give index to each element

    def transform(self, sentList):
        n = len(sentList)
        m = len(self.feature_map)
        ngram_feature = np.zeros((n, m))
        for idx, sent in enumerate(sentList):
            if self.do_lower_case:
                sent = sent.lower()
            sent = sent.split(" ")
            for gram in self.ngram:
                for i in range(len(sent) - gram + 1):
                    feature = "_".join(sent[i:i + gram])
                    if feature in self.feature_map:#若单词或单词对出现在map中，则在对应位置标注1（独热编码）
                        ngram_feature[idx][self.feature_map[feature]] = 1
                        
        return ngram_feature

    def fit_transform(self, sentList):
        self.fit(sentList)
        return self.transform(sentList)


if __name__ == "__main__":
    gram = NGram((1, 2))
    sents = ["I love you", "do you love yourself"]
    feature = gram.fit_transform(sents)
    print(gram.feature_map)
    print(feature)
