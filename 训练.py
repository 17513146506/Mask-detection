#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 步骤
# 1.读取NPZ文件
# 2.onehot 独热编码
# 3.分为train和test数据
# 4.搭建CNN模型
# 5.训练模型
# 6.保存模型


# In[2]:


## 1.读取NPZ文件


# In[3]:


import numpy as np


# In[4]:


arr = np.load('./data/imageData.npz')


# In[5]:


img_list = arr['arr_0']
label_list =arr['arr_1']


# In[6]:


img_list.shape,label_list.shape


# ## 2.onehot 独热编码

# In[7]:


np.unique(label_list)


# In[8]:


from sklearn.preprocessing import OneHotEncoder


# In[9]:


# 实例化
onehot = OneHotEncoder()


# In[10]:


# 编码
y_onehot =onehot.fit_transform(label_list.reshape(-1,1))


# In[11]:


y_onehot_arr = y_onehot.toarray()


# In[12]:


y_onehot_arr


# ## 3.分为train和test数据

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(img_list,y_onehot_arr,test_size=0.2,random_state=42)


# In[15]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# ## 4.搭建CNN模型

# In[16]:


# pip install --upgrade tensorflow
# pip install tensorflow-gpu==版本号  # GPU


# In[17]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential


# In[18]:


# 搭建模型


# ![](./cnn.png)

# In[19]:


model = Sequential([
    layers.Conv2D(16,3,padding='same',input_shape=(100,100,3),activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(166,activation='relu'),
    layers.Dense(22,activation='relu'),
    layers.Dense(3,activation='sigmoid')
])


# In[20]:


# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])


# In[21]:


# 预览模型
model.summary()


# ## 5.训练模型

# In[22]:


history = model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=30,epochs=15)


# In[23]:


# 查看训练效果


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


history_pd = pd.DataFrame(history.history)


# In[26]:


history_pd


# In[27]:


# 查看损失
plt.plot(history_pd['loss'])
plt.plot(history_pd['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train set','test set'],loc='upper right')
plt.show()


# In[28]:


# 查看准确率
plt.plot(history_pd['accuracy'])
plt.plot(history_pd['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train set','test set'],loc='upper right')
plt.show()


# ## 6.保存模型

# In[29]:


model.save('./data/face_mask_model')

