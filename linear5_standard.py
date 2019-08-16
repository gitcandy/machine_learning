import numpy as np
from math import sqrt

def mean_squared_error(y_test,y_predict):
    assert len(y_test)==len(y_predict),"the size of y_test must be equal to y_predict"
    return  sum((y_test-y_predict)**2)/len(y_test)

def root_mean_squared_error(y_test,y_predict):
    return  sqrt(mean_squared_error(y_test,y_predict))

def mean_absolute_error(y_test,y_predict):
    assert len(y_test) == len(y_predict), "the size of y_test must be equal to y_predict"
    return np.sum(np.absolute(y_predict-y_predict))/len(y_test)
def r2_squared(y_test,y_predict):
    """计算y_test和y_predict之间的R square"""
    return  1-mean_squared_error(y_test,y_predict)/np.var(y_test)