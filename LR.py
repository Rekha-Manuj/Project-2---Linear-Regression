# -*- coding: utf-8 -*-
"""
Congratulations! You just got some contract work with an Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to help them figure it out! Let's get started!

Just follow the steps below to analyze the customer data (it's fake, don't worry I didn't give you real credit card numbers or emails.

"""

Created on Fri Feb 11 21:47:45 2022

@author: REKHA
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import csv

df=pd.read_csv('Employer.csv')
df.info()
df.head()
df.describe()

sns.jointplot(x="Time on App",y="Yearly Amount Spent",kind='hex',data=df)
sns.jointplot(x="Time on App",y="Length of Membership",kind="hex",data=df)
sns.pairplot(df)


# create a linear model plot
sns.regplot(x="Yearly Amount Spent",y="Length of Membership",data=df)
 

# Training and Testing Data
# Now that we've explored the data a bit, 
#let's go ahead and split the data into training and testing sets. 
#Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column.

from sklearn.model_selection import train_test_split
y=df['Yearly Amount Spent'] # it will always be target variable or dependent variable
x=df[[  'Avg. Session Length', 'Time on App',
      'Time on Website', 'Length of Membership']] # independent features
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=12)


#Training the model
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
lm.coef_


#Predicting Test Data
prediction=lm.predict(X_test)
sns.scatterplot(y_test,prediction)
from sklearn.metrics import r2_score
r2_score(y_test,prediction)


#Evaluating the model

from sklearn import metrics
print('MAE',metrics.mean_absolute_error(y_test,prediction))
print('MSE',metrics.mean_squared_error(y_test,prediction))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


#Residuals
sns.distplot(y_test.prediction)


#Conclusion
pd.DataFrame(lm.coef_,x.columns,columns=['Coeffs'])


