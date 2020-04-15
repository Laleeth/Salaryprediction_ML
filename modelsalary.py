<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:55:07 2020

@author: ImLaleeth
"""
#importing all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('C:/Users/ImLaleeth/Desktop/Data Science/salary_predict_dataset.csv')
df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(),inplace=True)
df['interview_score'].fillna(df['interview_score'].mean(),inplace=True)

def convert_to_int(word):
    word_dict={'one':1,'two':2,'three':3,'four':4,'five':5,
               'six':6,'seven':7,'eight':8,'nine':9,
               'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fifteen':15,'zero':0,0:0}
    return word_dict[word]

X=df.iloc[:,:3]
y=df.iloc[:,-1]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))
X

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor,open('modelsalary.pkl','wb'))

model=pickle.load(open('modelsalary.pkl','rb'))
=======
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:55:07 2020

@author: ImLaleeth
"""
#importing all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('C:/Users/ImLaleeth/Desktop/Data Science/salary_predict_dataset.csv')
df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(),inplace=True)
df['interview_score'].fillna(df['interview_score'].mean(),inplace=True)

def convert_to_int(word):
    word_dict={'one':1,'two':2,'three':3,'four':4,'five':5,
               'six':6,'seven':7,'eight':8,'nine':9,
               'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fifteen':15,'zero':0,0:0}
    return word_dict[word]

X=df.iloc[:,:3]
y=df.iloc[:,-1]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))
X

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor,open('modelsalary.pkl','wb'))

model=pickle.load(open('modelsalary.pkl','rb'))
>>>>>>> a06de1b91a1094879339d2fadcc0a7964bfe7327
print(model.predict([[5,4,9]]))