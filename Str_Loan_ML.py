#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction using Machine Learning

# ### In this project, we will predict whether a customer will get the loan from bank or not.

# Following Factors are:
# 1. Gender
# 2. Education
# 3. Marrital status
# 4. Loand Amount
# 5. Credit History
# 6. Account Balance
# 7. Property Area
# 8. Credit History
# 9. Dependants
# 10. Self Employment Status
# 
# There are more factors also, let see in this notebook

# In[3]:


import pandas as pd
import numpy as np
import os
os.chdir("C:\Users\nouri\Loan Data")


# In[4]:


## Pandas
## Numpy
## SKlearn
## Matplotlib


# In[5]:


train=pd.read_csv('./Loan_Data/train.csv')
train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})


# ## Check the missing Values in data

# In[4]:


train.isnull().sum()


# ## Preprocessing on the data

# In[5]:


Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv('./Loan_Data/test.csv')
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


data.Dependents.dtypes


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# ## Label ENcode

# In[11]:


## Label encoding for gender
data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()


# In[12]:


## Let's see correlations
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[13]:


## Labelling 0 & 1 for Marrital status
data.Married=data.Married.map({'Yes':1,'No':0})


# In[14]:


data.Married.value_counts()


# In[15]:


## Labelling 0 & 1 for Dependents
data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})


# In[16]:


data.Dependents.value_counts()


# In[17]:


## Let's see correlations for it
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[18]:


## Labelling 0 & 1 for Education Status
data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})


# In[19]:


data.Education.value_counts()


# In[20]:


## Labelling 0 & 1 for Employment status
data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})


# In[21]:


data.Self_Employed.value_counts()


# In[22]:


data.Property_Area.value_counts()


# In[23]:


## Labelling 0 & 1 for Property area
data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})


# In[24]:


data.Property_Area.value_counts()


# In[25]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[26]:


data.head()


# In[27]:


data.Credit_History.size


# ## It's time to fill the missing values

# In[28]:


data.Credit_History.fillna(np.random.randint(0,2),inplace=True)


# In[29]:


data.isnull().sum()


# In[30]:


data.Married.fillna(np.random.randint(0,2),inplace=True)


# In[31]:


data.isnull().sum()


# In[32]:


## Filling with median
data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)


# In[33]:


## Filling with mean
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)


# In[34]:


data.isnull().sum()


# In[35]:


data.Gender.value_counts()


# In[36]:


## Filling Gender with random number between 0-2
from random import randint 
data.Gender.fillna(np.random.randint(0,2),inplace=True)


# In[37]:


data.Gender.value_counts()


# In[38]:


## Filling Dependents with median
data.Dependents.fillna(data.Dependents.median(),inplace=True)


# In[39]:


data.isnull().sum()


# In[40]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[41]:


data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)


# In[42]:


data.isnull().sum()


# In[43]:


data.head()


# In[44]:


## Dropping Loan ID from data, it's not useful
data.drop('Loan_ID',inplace=True,axis=1)


# In[45]:


data.isnull().sum()


# In[46]:


data.head()


# ## Split the Data into X & Y

# In[47]:


train_X=data.iloc[:614,] ## all the data in X (Train set)
train_y=Loan_status  ## Loan status will be our Y


# In[48]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=0)


# In[49]:


#sc_f = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
#sc_f = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
train_X.head()


# In[50]:


# train_X.head()


# In[51]:


test_X.head()


# ## Using Different types of Machine Learning Model

# In[52]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ## Fit the all ML Models

# In[53]:


models=[]
models.append(("Logistic Regression",LogisticRegression()))
models.append(("Decision Tree",DecisionTreeClassifier()))
models.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis()))
models.append(("Random Forest",RandomForestClassifier()))
models.append(("Support Vector Classifier",SVC()))
models.append(("K- Neirest Neighbour",KNeighborsClassifier()))
models.append(("Naive Bayes",GaussianNB()))


# In[54]:


scoring='accuracy'


# In[55]:


from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
result=[]
names=[]


# In[57]:


for name,model in models:
    kfold=KFold(n_splits=10,random_state=0)
    cv_result=cross_val_score(model,train_X,train_y,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    print(model)
    print("%s %f" % (name,cv_result.mean()))


# In[68]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

LR=LogisticRegression()
LR.fit(train_X,train_y)
pred=LR.predict(test_X)
print("Model Accuracy:- ",accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[69]:


print(pred)


# In[66]:


X_test=data.iloc[614:,] 
# X_test[sc_f]=SC.fit_transform(X_test[sc_f])


# In[67]:


X_test.head()


# In[63]:


prediction = LR.predict(X_test)


# In[70]:


print(prediction)


# In[64]:


## TAken data from the dataset
t = LR.predict([[0.0,	0.0,	0.0,	1,	0.0,	1811,	1666.0,	54.0,	360.0,	1.0,	2]])


# In[65]:


print(t)


# In[65]:


import pickle
# now you can save it to a file
file = './Model/ML_Model1.pkl'
with open(file, 'wb') as f:
    pickle.dump(svc, f)


# In[66]:


with open(file, 'rb') as f:
    k = pickle.load(f)


# In[68]:


cy = k.predict([[0.0,	0.0,	0.0,	1,	0.0,	4230,	0.0,	112.0,	360.0,	1.0,	1]])
print(cy)


# In[ ]:




