import numpy as np
from data_preprocess import read_data
from feature_extraction import BagOfWord, NGram
from softmax_regerssion import SoftmaxRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 读入数据
    X_data, y_data = read_data("dataset/train.tsv")

    
    X_data = X_data[:1000]
    y_data = y_data[:1000]
    y = np.array(y_data).reshape(len(y_data), 1)

    # 数据集划分
    bag_of_word_model = BagOfWord(do_lower_case=True)
    ngram_model = NGram(ngram=(1, 2), do_lower_case=True)
    X_Bow = bag_of_word_model.fit_transform(X_data)
    X_Gram = ngram_model.fit_transform(X_data)
    print("Bow shape", X_Bow.shape)
    print("Gram shape", X_Gram.shape)

    X_train_Bow, X_test_Bow, y_train_Bow, y_test_Bow = train_test_split(X_Bow, y, test_size=0.1, random_state=42, stratify=y)
    X_train_Gram, X_test_Gram, y_train_Gram, y_test_Gram = train_test_split(X_Gram, y, test_size=0.1, random_state=42, stratify=y)
    # 训练模型 不同特征的差别
    epoch = 50
    bow_learning_rate = 0.1
    gram_learning_rate = 0.1

    # 梯度下降
    model1 = SoftmaxRegression()
    history = model1.train(X_train_Bow, y_train_Bow, epoch=epoch, learning_rate=bow_learning_rate, print_loss_steps=epoch//10, update_strategy="stochastic")
    plt.plot(np.arange(len(history)), np.array(history))
    plt.show()
    print("Bow train {} test {}".format(model1.score(X_train_Bow, y_train_Bow), model1.score(X_test_Bow, y_test_Bow)))

    model2 = SoftmaxRegression()
    history = model2.train(X_train_Gram, y_train_Gram, epoch=epoch, learning_rate=gram_learning_rate, print_loss_steps=epoch//10, update_strategy="stochastic")
    plt.plot(np.arange(len(history)), np.array(history))
    plt.show()
    print("Gram train {} test {}".format(model2.score(X_train_Gram, y_train_Gram), model2.score(X_test_Gram, y_test_Gram)))