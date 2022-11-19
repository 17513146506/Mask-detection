#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 加载包


# In[3]:


import tensorflow as tf


# In[4]:


# 构造转换器
converter = tf.lite.TFLiteConverter.from_saved_model('./data/face_mask_model/')


# In[5]:


# 转换
tflite_model = converter.convert()


# In[7]:


# 保存lite


# In[8]:


with open('./data/face_mask.tflite','wb') as f:
    f.write(tflite_model)


# In[ ]:




