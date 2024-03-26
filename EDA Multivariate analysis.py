#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns


# In[3]:


tips = sns.load_dataset('tips')


# In[4]:


titanic = pd.read_csv("D:/ISIBangalore/MS-QMScourse/Datasets/titanic/train.csv")


# In[5]:


flights=sns.load_dataset('flights')


# In[6]:


iris=sns.load_dataset('iris')


# In[7]:


tips.head()


# In[8]:


iris.head()


# In[9]:


flights.head()


# In[10]:


titanic.head()


# # 1. Scatterplot(Numerical-Numerical)

# In[11]:


sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', style='smoker', size='size')


# # 2. Bar plot(Numerical-Categorical)

# In[12]:


titanic.head()


# In[13]:


sns.barplot(data=titanic, x='Pclass', y='Age', hue='Sex')


# In[14]:


sns.barplot(data=titanic, x='Pclass', y='Fare', hue='Sex')


# # 3. Box plot(Numerical-Categorical)

# In[15]:


sns.boxplot(data=titanic, x='Sex', y='Age', hue='Pclass')


# In[16]:


sns.boxplot(data=titanic, x='Sex', y='Age', hue='Survived')


# # 4. Dist plot(Numerical-Categorical)

# In[17]:


sns.kdeplot(data=titanic, x='Age', hue='Sex')


# # 5. HeatMap(categorical- categorical)

# In[18]:


titanic.head(3)


# In[20]:


sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']))


# In[25]:


(titanic.groupby('Pclass').mean()['Survived']*100).plot(kind='bar')


# In[27]:


(titanic.groupby('Sex').mean()['Survived']*100).plot(kind='bar')


# In[29]:


(titanic.groupby('Embarked').mean()['Survived']*100).plot(kind='bar')


# # 6. ClusterMap(categorical- categorical)

# In[32]:


sns.clustermap(pd.crosstab(titanic['Parch'], titanic['Survived']))


# In[33]:


sns.clustermap(pd.crosstab(titanic['SibSp'], titanic['Survived']))


# # 7. Pairplot

# In[34]:


iris.head()


# In[37]:


sns.pairplot(iris, hue='species')


# # 8.Lineplot(Numerical-Numerical)

# In[38]:


flights.head()


# In[68]:


z = flights.groupby('year').sum(numeric_only=True)


# In[69]:


z


# In[70]:


sns.lineplot(z)


# In[76]:


(flights.pivot_table(values='passengers', index='month', columns='year'))


# In[75]:


sns.heatmap(flights.pivot_table(values='passengers', index='month', columns='year'))


# In[77]:


sns.clustermap(flights.pivot_table(values='passengers', index='month', columns='year'))


# In[ ]:




