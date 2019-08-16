##s使用网格搜索方式选择最佳p和n_neighbors
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

digits=datasets.load_digits()
X=digits.data
y=digits.target

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=666)

# sk_knn_clf=KNeighborsClassifier(n_neighbors=5,weights='uniform')
# sk_knn_clf.fit(x_train,y_train)
# print(sk_knn_clf.score(x_test,y_test))


knn_clf=KNeighborsClassifier()

param_grid=[
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(1,11)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1,11)],
        'p':[i for i in range(1,6)]
    }
]

grid_search=GridSearchCV(knn_clf,param_grid)
grid_search.fit(x_train,y_train)
print(grid_search.score(x_test,y_test))