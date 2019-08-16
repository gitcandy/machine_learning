#简单的回归实现方法
import  numpy as np
class SimpleLinearRegression1:
    def __init__(self):
        """初始化regression模型"""
        self.a=None
        self.b=None
    def fit(self,x_train,y_train):
        """根据训练集x_train,y_train来训练回归模型"""
        assert x_train.ndim==1," the dimension must be one"
        assert len(x_train)==len(y_train),"the length of x_train must be equal to y_train"
        x_mean=np.mean(x_train)
        y_mean=np.mean(y_train)
        num=0.0
        d=0.0
        for x,y in zip(x_train,y_train):
            num+=(x-x_mean)*(y-y_mean)
            d+=(x-x_mean)**2
        self.a=num/d
        self.b=self.a*x_mean-y_mean
        return  self
    def predict(self,x_predict):
        """给定一个x的向量集，用来预测y的向量值"""
        assert  self.a is not None and self.b is not None,"a and be can not be None"
        assert  x_predict.ndim==1,"the dimension of x_predict must be one"
        return  np.array([self._predict(i) for i in x_predict])
    def _predict(self,x_single):
        """给定一个数，返回该数x_single的预测结果"""
        return  x_single*self.a+self.b
    def __repr__(self):
        return  "SimpleLinearRegression1()"


class SimpleLinearRegression2:
    def __init__(self):
        """初始化regression模型"""
        self.a=None
        self.b=None
    def fit(self,x_train,y_train):
        """根据训练集x_train,y_train来训练回归模型"""
        assert x_train.ndim==1," the dimension must be one"
        assert len(x_train)==len(y_train),"the length of x_train must be equal to y_train"
        x_mean=np.mean(x_train)
        y_mean=np.mean(y_train)
        num=0.0
        d=0.0
        num=(x_train-x_mean).dot(x_train-y_mean)
        d=(x_train-x_mean).dot(x_train-x_mean)
        self.a=num/d
        self.b=self.a*x_mean-y_mean
        return  self
    def predict(self,x_predict):
        """给定一个x的向量集，用来预测y的向量值"""
        assert  self.a is not None and self.b is not None,"a and be can not be None"
        assert  x_predict.ndim==1,"the dimension of x_predict must be one"
        return  np.array([self._predict(i) for i in x_predict])
    def _predict(self,x_single):
        """给定一个数，返回该数x_single的预测结果"""
        return  x_single*self.a+self.b
    def __repr__(self):
        return  "SimpleLinearRegression1()"