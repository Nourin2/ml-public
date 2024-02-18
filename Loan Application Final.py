#!/usr/bin/env python
# coding: utf-8

# # Building Frontend Of Application

# 1.Install necessary library
# 2.Create frontend app using streamlit

# In[1]:


pip install pyngrok


# In[2]:


pip install streamlit


# In[3]:


get_ipython().system('pyngrok version')


# In[4]:


get_ipython().system('streamlit version')


# Streamlit Librarires
# https://docs.streamlit.io/library/api-reference
# Neceassary Libraries for projects
# 

# In[5]:


get_ipython().system('pip install writefile')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', '#### Creating Python Script\nimport stramlit as st\n### function to define out app\ndef main():\n    ## page header\n    st.markdown(\'Loan Eligibility Checker\')\n    ## Loading data and creating boxes so where user will provide information ## we have four user \n    Gender = st.selectbox(\'Gender\',(\'Male\',\'Female\', \'Other\'))##creates a dropdown\n    Maritalstatus = st.selectbox(\'Maritalstatus\',(\'Married\', \'Unamrried\', \'Others\'))\n    MonthlyIncome = st.number_input(\'Monthly Income in Rupees\')\n    Loanamount = st.number_input(\'Loan Amount in Rupees\')\n    result = ""\n    ## If clicked make a prediction store it \n    if st.button(\'Check\'):\n        result = prediction(Gender, Maritalstatus, MonthyIncome, Loanamount)\n        #calling prediction function\n        st.success(f\'your loan is: {result}\') #display frontend\n        # defining \'prediction\' function => it will predict based on user input data\n        def predict(Gender, Maritalstatus, MonthlyIncome, Loanamount):\n            #building rules based to automate loan eligibility\n            ###### 3. building Rule based model to automate Loan eligibility ######\n    if (ApplicantIncome >=50000): # if loan ApplicantIncome greater or equall to 50k then approve \n      loan_status = \'Approved\'\n   # elif (LoanAmount < 500000): # elif loan amount less then 50 thou then approve\n       # loan_status = \'Approved\'\n    else:\n        loan_status = \'Rejected\'\n    return loan_status\n  \nif __name__ == \'__main__\':\n    main()\n    \n')


# # Deploying Application

# Streamlit host application on 8501 port bydefault
# This model is running locally for now
# To make it accessible to everyone(public):
# 
# using pyngrok lib

# In[8]:


pip install ngrok


# In[9]:


# making locally-hosted web application to be publicly accessible
from pyngrok import ngrok

public_url = ngrok.connect(port=8501)
public_url


# ngrok is a reverse proxy tool that opens secure tunnels from public URLs to localhost, perfect for exposing local web servers, building webhook integrations, enabling SSH access, testing chatbots, demoing from your own machine, and more, and its made even more powerful with native Python integration through pyngrok.

# # Steps to build Loan Eligibility model

# 1.Loading dataset, 
# 2.Pre-processing dataset,
# 3.Building Loan Prediction model,
# 4.Deploying machine learning model using Streamli

# In[10]:


import pandas as pd
import numpy as np


# In[11]:


import os

print(os.getcwd())


# In[12]:


# required libs
import pandas as pd
import numpy as np

import os
for dirname, _, filenames in os.walk('C:\Users\dell'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Loading Dataset

# In[13]:


# loading  dataset
data = pd.read_csv('loan_data.csv')
data.sample(5)


# In[14]:


data.shape


# In[15]:


# converting categories into numbers
data['Gender']= data['Gender'].map({'Male':0, 'Female':1})
data['Married']= data['Married'].map({'No':0, 'Yes':1})
data['Loan_Status']= data['Loan_Status'].map({'N':0, 'Y':1})


# In[16]:


# dependent and independent variables
X = data[['Gender','Married','ApplicantIncome','LoanAmount']]
y = data.Loan_Status


# Building Loan Prediction Model

# In[17]:


# importing ML model
from sklearn.linear_model import LogisticRegression

# training logistic regression model
model = LogisticRegression() 
model.fit(X,y)


# In[18]:


# saving model 
import pickle 
'''
Saving model as we will use it when neaded 
we need not to run model again and times
'''

pickle_out = open('classifier_model.pkl',mode ='wb')  # wb => write in binary mode
pickle.dump(model,pickle_out)  
pickle_out.close()


# # 4. Deploying machine learning model using Streamlit

# Building Frontend of application
# Loading and Pre-processing data
# Building Machine Learning model to automate Loan Eligibility
# Deploying application
# 1. Building Frontend of application
# 1.1. Installing Required Libraries
# 1.2. Creating Frontend of app using Streamlit

# In[19]:


# installing pyngrok
get_ipython().system('pip install -q pyngrok')


# In[20]:


# installing streamlit
get_ipython().system('pip install -q streamlit')


# Creating frontend of app using Streamlit

# In[21]:


get_ipython().run_cell_magic('writefile', 'app.py', '\n# importing required libraries\nimport pickle\nimport streamlit as st\n\n# loading the trained model\npath_model = \'./classifier_model.pkl\'\npickle_in = open(path_model,\'rb\')  # rb => read binary file\nclassifier = pickle.load(pickle_in) # loading model in variable\n\n# this is main function in which we define our app  \ndef main():       \n    # header of the page \n    html_temp = """ \n    <div style ="background-color:red;padding:13px"> \n    <h1 style ="color:white;text-align:center;">Check your Loan Eligibility</h1> \n    </div> \n    """\n    st.markdown(html_temp,unsafe_allow_html=True) \n\n    # creating boxes for user input => data required to make prediction \n    Gender = st.selectbox(\'Gender\',(\'Male\',\'Female\',\'Other\'))\n    Married = st.selectbox(\'Marital Status\',(\'Unmarried\',\'Married\',\'Other\')) \n    ApplicantIncome = st.number_input(\'Monthly Income in INR\') \n    LoanAmount = st.number_input(\'Loan Amount in INR\')\n    result =""\n      \n    # when \'Check\' will be clicked => make prediction and store it \n    if st.button(\'Check\'): \n        result = prediction(Gender,Married,ApplicantIncome,LoanAmount) \n        st.success(f\'Your loan is: {result}\') # display output to frontend\n \n# defining function which will make prediction using data which user inputs \ndef prediction(Gender,Married,ApplicantIncome,LoanAmount): \n\n    ########### 2. Loading and Pre-processing data ###########\n    # as input will be in object changing it into numbers\n    if Gender == \'Male\':\n        Gender = 0\n    else:\n        Gender = 1\n\n    if Married == \'Married\':\n        Married = 1\n    else:\n        Married = 0\n\n    ############ 3. Building ML model to automate Loan Eligibility  ###########\n    # prediction varible will have output as 0 or 1\n    prediction = classifier.predict([[Gender,Married,ApplicantIncome,LoanAmount]]) # use of previously made classifier varaible\n    \n    # if prediction is 0 reject loan else approve loan\n    if prediction == 0:\n        pred = \'Rejected\'\n    else:\n        pred = \'Approved\'\n    return pred # returning result\n     \nif __name__==\'__main__\': \n    main()\n')


# In[ ]:


# running app
get_ipython().system('streamlit run app.py')


# In[ ]:





# In[ ]:




