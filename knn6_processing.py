#尝试理解satandrScaleg的归一化处理思想
import  numpy as np
class StandarScale:
    def __init__(self):
        self.mean_=None
        self.scale_=None
    def fit(self,x):
        """根据传入训练的数据集求数据的均值和方差"""
        assert x.ndim==2,"the dimension of x must be 2"
        self.mean_=np.array([np.mean(x[:,i])  for i in range(1,x.shape[1])])
        self.scale_=np.array([np.std(x[:,i]) for i in range(1,x.shape[1])])
        return self
    def transformation(self,x):
        """对传入的数据进行均值方差归一化处理"""
        assert x.ndim==2,"the dimension of x must be 3"
        assert self.mean_ is not None and self.scale_ is not None,"the mean and scale is not None"
        assert  x.shape[1]==len(self.mean_),"the size of x must be equal to mean"
        resx=np.empty(shape=x.shape,dtype=float)
        for col in range(x.shape[1]):
            x[:,col]=(x[:,col]-self.mean_[col])/self.scale_[col]
        return resx


