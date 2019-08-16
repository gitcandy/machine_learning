import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
def train_test_split(X,Y,test_ratio=0.2,seed=None):
    '''将数据X和Y按照test_ration分分割成x_tain,y_train,x_test,y_test'''
    assert X.shape[0]==Y.shape[0], "the size of x must be equal to the size of y"
    assert 0.0<= test_ratio<=1.0,"test ratio must be valid"
    #seed的使用：保重随机的是一次性的，不会执行再发生变化
    if seed:
        np.random.seed()

    shuffle_indexs=np.random.permutation(len(X))
    test_size=int(len(X)*test_ratio)
    test_indexs=shuffle_indexs[:test_size]
    train_indexs=shuffle_indexs[test_size:]
    x_train=X[train_indexs]
    x_test=X[test_indexs]
    y_train=Y[train_indexs]
    y_test=Y[test_indexs]
    return  x_train,x_test,y_train,y_test

# iris=datasets.load_iris()
# X=iris.data
# Y=iris.target
# x_train,y_train,x_test,y_test=train_test_split(X,Y)
# print(x_train.shape[0],y_train.shape[0])















