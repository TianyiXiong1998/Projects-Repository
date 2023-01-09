import numpy as np


def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    z = np.exp(z)
    z /= np.sum(z, axis=1, keepdims=True)
    return z


class SoftmaxRegression:
    def __init__(self):
        self.num_of_class = None  # 类别数量
        self.n = None   # 数据个数
        self.m = None   # 数据维度
        self.weight = None  # 模型权重 shape (类别数，数据维度)
        self.learning_rate = None

    def train(self, X, y, learning_rate=0.01, epoch=10, num_of_class=5, print_loss_steps=-1, update_strategy="batch"):
        self.n, self.m = X.shape
        self.num_of_class = num_of_class
        self.weight = np.random.randn(self.num_of_class, self.m)
        self.learning_rate = learning_rate
        y_one_hot = np.zeros((self.n, self.num_of_class))
        for i in range(self.n):
            y_one_hot[i][y[i]] = 1
        loss_history = []

        for e in range(epoch):
            loss = 0
            if update_strategy == "stochastic":
                rand_index = np.arange(len(X))
                np.random.shuffle(rand_index)
                for index in list(rand_index):
                    Xi = X[index].reshape(1, -1)
                    
                    prob = Xi.dot(self.weight.T)
                    prob = softmax(prob).flatten()
                    
                    loss += -np.log(prob[y[index]])
                    self.weight += Xi.reshape(1, self.m).T.dot((y_one_hot[index] - prob).reshape(1, self.num_of_class)).T

            if update_strategy == "batch":
                prob = X.dot(self.weight.T)  
                prob = softmax(prob)

                for i in range(self.n):
                    loss -= np.log(prob[i][y[i]])

                # 书中给的损失函数
                weight_update = np.zeros_like(self.weight)
                for i in range(self.n):
                    weight_update += X[i].reshape(1, self.m).T.dot((y_one_hot[i] - prob[i]).reshape(1, self.num_of_class)).T
                self.weight += weight_update * self.learning_rate / self.n

            loss /= self.n
            loss_history.append(loss)
            if print_loss_steps != -1 and e % print_loss_steps == 0:
                print("epoch {} loss {}".format(e, loss))
        return loss_history

    def predict(self, X):
        prob = softmax(X.dot(self.weight.T))
        return prob.argmax(axis=1)

    def score(self, X, y):
        pred = self.predict(X)
        pred = pred.reshape(y.shape)
        correct = np.sum(pred.reshape(y.shape) == y) # count the correct predicted result.
        ratio = correct/y.shape[0]
        return ratio