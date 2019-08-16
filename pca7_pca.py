import numpy as np

class PCA:
    def __init__(self,n_components):
        """初始化PCA"""
        assert n_components>=1,"n_components must be valid"
        # 一共提取几个成分
        self.n_conmponents_=n_components
        # 提取的成分为什么
        self.componets=None

    def fit(self,x,eta=0.01,n_iters=1e4):
        """获得数据集x的前n个主成分"""
        assert self.n_conmponents_<=x.shape[1],"n_components must not be greater than the dimension of x"

        def demean(x):
            return x - np.mean(x, axis=0)

        def f(w, x):
            return np.sum((x.dot(w) ** 2)) / len(x)

        # 梯度函数
        def df(w, x):
            return x.T.dot(x.dot(w)) * 2. / len(x)

        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(x, initial_w, eta, n_iters=1e4, epsilon=1e-8):
            w = direction(initial_w)

            n_iter = 0
            while (n_iter < n_iters):
                gradient = df(w, x)
                last_w = w
                w = w + eta * w
                w = direction(w)  # 求单位方向

                if (np.abs(f(w, x) - f(last_w, x)) < epsilon):
                    break
            n_iter += 1
            return w
        x_pca=demean(x)
        self.componets=np.empty(shape=(self.n_conmponents_,x.shape[1]))
        for i in range(self.n_conmponents_):
            initial_w=np.random.random(x_pca.shape[1])
            w=first_component(x_pca,initial_w,eta,n_iters)
            self.componets[i,:]=w
            # 求下一个主成分
            x_pca=x_pca-x_pca.dot(w).reshape(-1,1)*w
        return  self
    def tranform(self,x):
        """将给定的x映射到各个主成分的分量中"""
        assert self.componets.shape[1]==x.shape[1],"dimension must be equal"
        return  x.dot(self.componets.T)
    def inverse_transform(self,x):
        """将给定的X反向映射会原来的空间"""
        assert x.shape[1]==self.componets.shape[1],"rols of x must be equal to cols of components"

        return x.dot(self.componets)
    def __repr__(self):
        return "PCA(n_components=%d)"% self.n_conmponents_
