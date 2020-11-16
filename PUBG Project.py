#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


# TASK 1

df = pd.read_csv("pubg.csv")

df


# In[8]:


#TASK 2

# checking datatype of all columns

data_type = df.dtypes
data_type


# In[9]:


# TASK 3

# DataFrame.describe() is used to give out the summary of all the numerical columns in a dataframe
num_summary = df.describe()
num_summary


# In[11]:


# TASK 4

avg = df['kills'].mean()
print("\nThe average person kills :", avg,"player")


# In[12]:


# TASK 5

nn_per = df["kills"].quantile(0.99)
print("\n99% of people have",nn_per,"kills")


# In[13]:


# TASK 6

most_kill = df["kills"].max()
print("\nThe most kill ever recorded are :",most_kill)


# In[14]:


# TASK 7

df.columns


# In[15]:


# TASK 8

sns.distplot( df['matchDuration'] );

# the match's duration is normally distributed


# In[16]:


# TASK 9

sns.distplot( df['walkDistance'] );

# Walk distance is positively skewed.


# In[17]:


# TASK 10

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('classic')
plt.figure()

# ploting for matchDuration
plt.subplot(2,1,1)
plt.plot(df["matchDuration"],"-")
plt.xlabel("Match Duration")

# ploting for walkDistance
plt.subplot(2,1,2)
plt.plot(df["walkDistance"],"--")
plt.xlabel("Walk Distance")


# In[18]:


# TASK 11

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('classic')
plt.figure(figsize=(10,5))

# ploting for matchDuration
plt.subplot(1,2,1)
plt.plot(df["matchDuration"])
plt.xlabel("Match Duration")

# ploting for walkDistance
plt.subplot(1,2,2)
plt.plot(df["walkDistance"])
plt.xlabel("Walk Distance")


# In[19]:


# TASK 12

sns.pairplot(df.head(700));


# In[20]:


#TASK 13

uni = pd.unique(df['matchType'])
print("\nUnique value in matchType is :",uni)
n_uni = len(uni)
print("\nCount of unique value in matchType is :",n_uni)


# In[5]:


#TASK 14

import seaborn as sns

sns.barplot(df['matchType'],df['killPoints'])
plt.xticks(rotation= 70);


# In[6]:


# TASK 15

sns.barplot(df['matchType'],df['weaponsAcquired'])
plt.xticks(rotation= 70);


# In[23]:


# TASK 16

cat_col = df.select_dtypes('category').columns 
cat_col


# In[7]:


# TASK 17

sns.boxplot(x='matchType', y='winPlacePerc', data=df)
plt.xticks(rotation= 70);


# In[8]:


# TASK 18

sns.boxplot(x='matchType', y='matchDuration', data=df)
plt.xticks(rotation= 70);


# In[26]:


# TASK 19

sns.boxplot( x='matchDuration', y='matchType',data=df);


# In[27]:


# TASK 20

df['KILL'] = df['headshotKills'] + df['teamKills'] + df['roadKills']
df['KILL']


# In[28]:


# TASK 21

df['winPlacePerc'].round(decimals=2)


# In[15]:


# TASK 22

mean = []
data = []

for i in range(100):
  
  for x in range(0,1001,50):
    
    mean1 = df['damageDealt'].head(x).mean()
    mean.append(mean1)
    print(mean1)
 


# In[ ]:




