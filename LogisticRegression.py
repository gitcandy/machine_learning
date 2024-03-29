import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self):
        # 初始化模型
        self.coef_=None
        self.intercept_=None
        self.theta_=None
    def sigmoid(self,x):
        return 1./(1.+np.exp(-x))

 # 利用梯度法进行线性回归预测
    def fit(self,x_train,y_train,eta=0.01,n_iters=1e4):

        assert x_train.shape[0]==y_train.shape[0],"the size of x_train must be equal to y_train"

        def J(theta, x_b, y):

            y_hat = self.sigmoid(X_b.dot(theta))
            try:
                return - np.sum([y*np.log(y_hat) + (1-y)*np.log(1-y_hat)]) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(self.sigmoid(X_b.dot(theta)) - y) / len(y)



        def gradient_decent(x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            n_iter = 0
            while (n_iter < n_iters):
                gradient = dJ(theta, x_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (np.abs(J(theta, x_b, y) - J(last_theta, x_b, y)) < epsilon):
                    break
                n_iter += 1
            return theta

        X_b = np.hstack([np.ones([len(x_train), 1]), x_train])
        initial_theta = np.zeros(X_b.shape[1])
        self.theta_=gradient_decent(X_b,y_train,initial_theta,eta)
        self.coef_=self.theta_[1:]
        self.intercept_=self.theta_[0]
        return  self


    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self.sigmoid(X_b.dot(self.theta_))




    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    def __prepare__(metacls, name, bases):
        return "LogisticRegression"

