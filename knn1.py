
# coding: utf-8

# In[29]:


import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter


# In[2]:


rawdata_x=[[3.39,2.33],
           [3.11,1.78],[1.34,3.76],[3.58,4.67],[2.28,2.86],[7.42,4.69],[5.74,3.53],[9.17,2.51],[7.79,3.42],[7.93,0.79]]
rawdata_y=[0,0,0,0,0,1,1,1,1,1]
rawdata_x,rawdata_y


# In[3]:


x_train=np.array(rawdata_x)
y_train=np.array(rawdata_y)
x_train,y_train


# In[14]:


plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],color="green",marker="+")
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],color="blue")
plt.scatter(x[0],x[1],color="red",marker="*")
plt.show()


# In[7]:


x=np.array([8.08,3.36])


# ### KNN算法的过程

# In[16]:


distances=[]
for xtrain in x_train:
    d=sqrt(np.sum((xtrain-x)**2))
    distances.append(d)


# In[19]:


distances=[sqrt(np.sum((xtrain-x)**2)) for xtrain in x_train]


# In[20]:


distances


# In[21]:


np.argsort(distances)


# In[24]:


nearnest=np.argsort(distances)
nearnest


# In[25]:


nearnest


# In[26]:


k=6


# In[27]:


topk_y=[y_train[i] for i in nearnest[:k] ]
topk_y


# In[31]:


Counter(topk_y)


# In[33]:


votes=Counter(topk_y)


# In[34]:


votes.most_common(2)


# In[35]:


votes.most_common(1)[0][0]


# In[37]:


predict_y=votes.most_common(1)[0][0]


# In[38]:


predict_y

