import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
from  knn4 import  accuracy_score

def kNN_classify(k,x_train,y_train,x):
    assert 1<= k <= x_train.shape[0],"k must be valid"
    assert  x_train.shape[0]==y_train.shape[0],"the size of x_train must equal to the size of y_train"
    assert  x_train.shape[1]==x.shape[0],"the feature number of x must be equal to x_train"
    distances=[sqrt(np.sum((xtrain-x)**2)) for xtrain in x_train]
    nearnest=np.argsort(distances)
    topk_y=[y_train[i] for i in nearnest[:k]]
    votes=Counter(topk_y)
    return votes.most_common(1)[0][0]


class KnnClassifier:
    def __init__(self,k):
        """初始化KNN分类器"""
        assert  k>=1,"k must be valid"
        self.k=k
        self.x_train=None
        self.y_train=None
    def fit(self,x_train,y_train):
        """根据训练数据集x_train,y_train训练KNN分类器"""
        assert  x_train.shape[0]==y_train.shape[0],"the size of the x_train must be equal to y_train"
        assert  self.k<=x_train.shape[0],"the size of x_train must be at least k"
        self.x_train=x_train
        self.y_train=y_train
        return  self
    def predict(self,x_predict):
        assert self.x_train is not None and self.x_train is not None,"the data is not none"
        assert  x_predict.shape[1]==self.x_train.shape[1]
        y_predict=[self._predict(x) for x in x_predict]
        return  np.array(y_predict)
    def  _predict(self,x):
        assert x.shape[0]==self.x_train.shape[1],"the feature of x must be equal to x_train"
        distances=[sqrt(np.sum((x-x_train )**2)) for x_train in self.x_train]
        nearnest=np.argsort(distances)
        took_y=[self.y_train[i] for i in nearnest[:self.k]]
        votes=Counter(took_y)
        return  votes.most_common(1)[0][0]
    def score(self,x_test,y_test):
        '''根据当前x_test和y_test来预测模型的准确度'''
        y_predict=self.predict(x_test)
        return  accuracy_score(y_test,y_predict)

    def _prer_(self):
        print("KNN number is ",self.k)





















