#!/usr/bin/env python
# coding: utf-8

# # Mobile Price EDA and Prediction

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train=pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Mobile-Price-Prediction-using-ML-Algorithm\\train.csv")
test=pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Mobile-Price-Prediction-using-ML-Algorithm\\test.csv")


# In[3]:


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


test.drop('id',axis=1,inplace=True)


# In[7]:


test.head()


# In[8]:


sns.countplot(train['price_range'])


# In[9]:


train.shape,test.shape


# In[10]:


train.isnull().sum()


# In[11]:


train.info()


# In[12]:


test.info()


# In[13]:


train.describe()


# In[14]:


plt.xlabel("Price Range")
plt.ylabel("Ram")
plt.bar(train['price_range'],train['ram'] )


# In[15]:


train.plot(x='price_range',y='battery_power',kind='scatter')
plt.show()
plt.bar(train['price_range'],train['battery_power'] )


# In[16]:


train


# In[17]:


#train.plot(x='price_range',y='fc',kind='scatter')
##plt.show()
plt.xlabel("FC")
plt.ylabel("Price Range")
plt.bar(train['price_range'],train['fc'] )


# In[18]:


train.plot(x='price_range',y='n_cores',kind='scatter')
plt.show()


# In[19]:


import seaborn as sns
plt.figure(figsize=(20,20))
sns.heatmap(train.corr(),annot=True,cmap=plt.cm.Accent_r)
plt.show()


# In[20]:


train.plot(kind='box',figsize=(20,10))


# In[21]:


X = train.drop('price_range',axis=1)
y = train['price_range']


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.1,random_state=101)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test = sc.transform(test)


# In[24]:


X_train


# In[25]:


X_test


# In[26]:


test


# In[27]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train , Y_train)


# In[28]:


pred = dtc.predict(X_test)
pred


# In[42]:


from sklearn.metrics import accuracy_score, confusion_matrix
dtc_acc = accuracy_score(pred,Y_test)
print(dtc_acc)
print(confusion_matrix(pred,Y_test))


# In[30]:


from sklearn.svm import SVC
knn=SVC()
knn.fit(X_train,Y_train)


# In[31]:


pred1 = knn.predict(X_test)
pred1


# In[32]:


from sklearn.metrics import accuracy_score
svc_acc = accuracy_score(pred1,Y_test)
print(svc_acc)
print(confusion_matrix(pred1,Y_test))


# In[33]:


from sklearn.linear_model import LogisticRegression  # its a classification
lr=LogisticRegression()
lr.fit(X_train,Y_train)


# In[34]:


pred2 = lr.predict(X_test)
pred2


# In[35]:


from sklearn.metrics import accuracy_score
lr_acc = accuracy_score(pred2,Y_test)
print(lr_acc)
print(confusion_matrix(pred2,Y_test))


# In[36]:


plt.bar(x=['dtc','svc','lr'],height=[dtc_acc,svc_acc,lr_acc])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.show()


# In[37]:


lr.predict(test)


# In[38]:


train[train['price_range'] == 1].count()


# In[39]:


#Major Steps:
#Data Collection
#EDA
#Training
#Testing
#User Inputs
#Deployment


# In[40]:


#Graphs and relationship is well analysed.
#All price range have the same count in the dataset
# Range of Ram is increasing with the price range according to the  bar graph
#Corelation matrix with the range corelation is observed.
#Model is trained using Decision tree classifier, KNN, Logistic regression. 
#Aim is to make a high accurate model.


# In[ ]:





# In[ ]:




