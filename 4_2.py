
# coding: utf-8

# In[3]:


import sys
sys.path.append('E:\python\machine_learning')


# In[9]:


get_ipython().run_line_magic('run', 'E:/python/machine_learning/knn2.py')


# In[18]:


rawdata_x=[[3.39,2.33],
           [3.11,1.78],[1.34,3.76],[3.58,4.67],[2.28,2.86],[7.42,4.69],[5.74,3.53],[9.17,2.51],[7.79,3.42],[7.93,0.79]]
rawdata_y=[0,0,0,0,0,1,1,1,1,1]
x_train=np.array(rawdata_x)
y_train=np.array(rawdata_y)

x=np.array([8.08,3.36])


# In[42]:


x=np.array([8.02,1])
x


# In[35]:


kNN_classify(6,x_train,y_train,x)


# ## 使用scikit-learn中的KNN

# In[13]:


from sklearn.neighbors import KNeighborsClassifier


# In[36]:


Knn_classifer=KNeighborsClassifier(n_neighbors=6)


# In[37]:


Knn_classifer.fit(x_train,y_train)


# In[24]:


x.shape


# In[43]:


x=x.reshape(1,-1)


# In[44]:


Knn_classifer.predict(x)


# ## 重新整理KNN算法
