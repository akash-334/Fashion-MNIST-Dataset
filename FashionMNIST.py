#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[95]:


import tensorflow as tf
from tensorflow import keras


# In[96]:


fashion=keras.datasets.fashion_mnist


# In[97]:


(x_train_full,y_train_full),(x_test,y_test)=fashion.load_data()


# In[98]:


plt.imshow(x_train_full[1])


# In[99]:


class_names=["T-shit","trouser","Pullover","Dress","Coat","Sandal","shirt","Sneaker","bag","ankle boot"]


# In[100]:


class_names[y_train_full[1]]


# In[55]:


plt.imshow(x_test[1])


# In[102]:


x_train_n=x_train_full/255.
x_test_n=x_test/255.


# In[103]:


x_valid,x_train=x_train_n[:5000],x_train_n[5000:]
y_valid,y_train=y_train_full[:5000],y_train_full[5000:]
y_train.shape


# In[104]:


x_test=x_test_n


# In[105]:


np.random.seed(42)
tf.random.set_seed(42)


# In[106]:


model=keras.models.Sequential()


# In[107]:


model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation="Softmax"))


# In[108]:


model.summary()


# In[109]:


w,b=model.layers[2].get_weights()
print(w,b)


# In[110]:


w.shape


# In[111]:


b.shape


# In[112]:


model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])


# In[113]:


model_history=model.fit(x_train,y_train,validation_data=(x_valid,y_valid),epochs=30)


# In[114]:


model.evaluate(x_test,y_test)


# In[115]:


model.predict(x_test)


# In[116]:


y_prob=model.predict(x_test)
y_prob.round(2)


# In[117]:


y_pred=np.argmax(y_prob,axis=1)
y_pred


# In[118]:


y_test


# In[119]:


from sklearn.metrics import accuracy_score


# In[120]:


accuracy_score(y_pred,y_test)


# In[121]:


plt.imshow(x_test[1])


# In[122]:


plt.imshow(x_test[2])


# In[125]:


plt.imshow(x_test[4])


# In[101]:


y_train_full[0]


# In[129]:


plt.imshow(x_train[54989])


# In[130]:


y_train[54989]


# In[132]:


class_names[1]


# In[ ]:




