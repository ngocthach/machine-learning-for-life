#!/usr/bin/env python
# coding: utf-8

# # Import

# In[41]:


import random
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Embedding
from keras.layers import LSTM
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# ## Process Data

# In[25]:


df_train = pd.read_csv('data/train.csv', sep=',')
df_test = pd.read_csv('data/test.csv', sep=',')
df_train.head()


# In[26]:


df_test.head()


# In[6]:


target_map = {1:'pop', 0:'rap'}
df_train['genre'] = df_train['class'].map(target_map)
df_train.head()


# In[ ]:


df_train['lyric']=df_train['lyric'].str.replace(',','')
df_test['lyric']=df_test['lyric'].str.replace(',','')


# In[27]:


train_text = df_train.lyric
train_label = df_train['class']

test_text = df_test.lyric


# ## Build Features

# In[13]:


token = Tokenizer(num_words=4000)
token.fit_on_texts(train_text)
x_train_seq = token.texts_to_sequences(train_text)
x_train = pad_sequences(x_train_seq, maxlen = 400)
x_train = np.array(x_train)
y_train = np.array(train_label).reshape(-1,1)


# In[28]:


x_test_seq = token.texts_to_sequences(test_text)
x_test = pad_sequences(x_test_seq, maxlen = 400)
x_test = np.array(x_test)


# ## Train Model

# In[14]:


model = Sequential()
model.add(Embedding(output_dim=32,input_dim = 4000,input_length=400))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))


# In[15]:


print(model.summary())


# In[33]:


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
train_history = model.fit(x_train,y_train,batch_size=400,epochs=10,verbose=2,
                         validation_split=0.2)train_history = model.fit(x_train,y_train,batch_size=400,epochs=10,verbose=2,
                         validation_split=0.2)


# In this example, y_true is a 1D array of true binary labels (0 or 1), and y_pred is a 1D array of predicted probabilities, which should be in the range [0, 1]. The roc_auc_score() function calculates the AUC by computing the ROC (Receiver Operating Characteristic) curve and then integrating the area under the curve. The resulting AUC value is a number between 0 and 1, where higher values indicate better performance.
# 
# Note that the roc_auc_score() function assumes that the predicted probabilities are for the positive class (i.e., class 1). If your model is a binary classifier that predicts the probability of the negative class (i.e., class 0), you can invert the predictions by subtracting them from 1 before passing them to roc_auc_score()

# ## Evaluate

# ### AUC

# In[45]:


prediction = model.predict(x_test)

# you can choose a threshold to convert it to labels
threshold = 0.5
predicted_labels = (prediction > threshold).astype(int)

# after you label for this dataset by manual, you have a real label, it can used to evaluate your model
y_true = [random.randint(0, 1) for _ in prediction]  
auc = roc_auc_score(y_true, prediction)
print('AUC:', auc)


# ### Loss & Accuracy

# In[46]:


# or use directly model to evaluate
scores = model.evaluate(x_test, predicted_labels, verbose=0)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])


# In[ ]:




