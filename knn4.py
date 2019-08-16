import  numpy as np
def accuracy_score(y_true,y_predict):
    '''计算y_true 和y_predict的准确率'''
    assert  y_predict.shape[0]==y_true.shape[0],"y_predict size must be equal to y_true size"
    accuracy=sum(y_true==y_predict)/len(y_true)
    return  accuracy