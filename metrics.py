import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# 求混淆矩阵
def Confusion_Matix(y_true,y_predict):
    return np.array([
        [TN(y_true,y_predict),FN(y_true,y_predict)],
        [FP(y_true,y_predict),TP(y_true,y_predict)]
    ])
#求精准度
def predicision_score(y_true,y_predict):
    tp=TP(y_true,y_predict)
    fp=FP(y_true,y_predict)
    try:
        return tp/(tp+fp)
    except:
        return 0.0
#求召回率
def recall_score(y_true,y_predict):
    tp=TP(y_true,y_predict)
    fn=FN(y_true,y_predict)
    try:
        return tp/(tp+fn)
    except:
        return 0.0

def TN(y_true,y_predict):
    assert len(y_true)==len(y_predict),"the length of y_true must be equal to y_predict"
    return np.sum((y_true==0)&(y_predict==0))
def TP(y_true,y_predict):
    assert len(y_true)==len(y_predict),"the length of y_true must be equal to y_predict"
    return np.sum((y_true==1)&(y_predict==1))
def FP(y_true,y_predict):
    assert len(y_true)==len(y_predict),"the length of y_true must be equal to y_predict"
    return np.sum((y_true==0)&(y_predict==1))
def TN(y_true,y_predict):
    assert len(y_true)==len(y_predict),"the length of y_true must be equal to y_predict"
    return np.sum((y_true==0)&(y_predict==0))

#用于描绘roc曲线的两个指标

def FPR(y_test,y_predict):
    fp=FP(y_test,y_predict)
    tn=TN(y_test,y_predict)
    try:
        return fp/(fp+tn)
    except:
        return 0.0


def TPR(y_test,y_predict):
    tp=TP(y_test,y_predict)
    fp=FP(y_test,y_predict)
    try:
        return tp/(tp+fp)
    except:
        return 0.0