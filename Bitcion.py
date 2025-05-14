#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r'D:\BTC-USD.csv')
df.head()


# In[3]:


df=df.dropna(axis=0,how="any")


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()


# In[8]:


df[df['Close'] == df['Adj Close']].shape, df.shape


# In[9]:


df = df.drop(['Adj Close'], axis=1)


# In[10]:


df.isnull().sum()


# In[19]:


plt.features = ['Open', 'High', 'Low', 'Close']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,2,i+1)
sb.distplot(df[col])
plt.show()


# In[12]:


plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,2,i+1)
sb.boxplot(df[col])
plt.show()


# In[13]:


splitted = df['Date'].str.split('-', expand=True)

df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

df.head()


# In[22]:


data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
plt.subplot(2,2,i +2)
data_grouped[col].plot.bar()
plt.show()


# In[15]:


df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()


# In[16]:


df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# In[17]:


plt.pie(df['target'].value_counts().values, 
labels=[0, 1], autopct='%1.1f%%')
plt.show()


# In[18]:


plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

