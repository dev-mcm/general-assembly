""" 
DAY 2

Starting Exercise
Load in housing csv
Run linear regression on X, Y

 Use all columns for x

Pull up coefficients, R2 Value

"""

#load libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib as mpl


#initialize import model
lreg = LinearRegression()

#import housing data 
df = pd.read_csv('data/housing.csv')

X = df.iloc[:,0:-1]

X -= X.mean()
X /= X.std()

y = df['PRICE']

lreg.fit(X,y)

lreg.coef_

coeff_table = {
    'Variable': X.columns,
    'Coefficient': lreg.coef_
}

coeffs = pd.DataFrame(coeff_table)

coeffs.sort_values(by='Coefficient', ascending=False)

lreg.score(X, y)

"""
R^2 represents the "explanatory power" of the model
Compares the accuracy of the model that we have with a "naive baseline", like if you were predicting the average value of Y

Remember Cost = EPSILON(Answer - Guess)^2


R^2 only works for linear regression, for classification there is accuracy, F1-score and ROC score (F1 and ROC might be the same thing)
"""

#Get the predictive value of Y for each set of X values
lreg.predict(X)

#Add prediction ot dataframe
df['PREDICTION'] = lreg.predict(X)
df.head()

our_model = np.sum((df['PRICE'] - df['PREDICTION'])**2)

#Average value of y
df['PRICE'] - y.mean()

naive_model = np.sum((df['PRICE'] - y.mean())**2)

1 - (our_model/naive_model)

import seaborn as sns

sns.regplot(x='PRICE',y='PREDICTION',data=df)

df.head()

#messing around with different plotting
df['Mean'] = df['PRICE'].mean()

df['squared_error'] = (df['PRICE'] - df['PREDICTION'])**2

df['error'] = (df['PRICE'] - df['PREDICTION'])

sns.regplot(x='PRICE',y='Mean', data=df)

sns.regplot(x='PRICE',y='squared_error', data=df)
sns.regplot(x='PRICE',y='error', data=df)


""" 
Bias vs. Variance trade off

Bias refers to when your dad doesn't adequately capture the information necessary to solve your problem
-- rule of thumb would be if your input data isn't enough information for a subject matter expert to make an educated guess. E.g. if you were trying to predict housing price based off of distance from highway + number of bathrooms
(underfit)

Variance refers to when your model mistakenly interprets random correlations as meaningful
This is sort of the opposite issue where the model interprets random correlations as meaningful
(overfit)

Cross validation tries to address these issues
To do this we split our data into two separate parts -- Training set and test set
Typically standard is 80% training set 20% test set
"""

from sklearn.model_selection import train_test_split 

#create train and test data sets w/20% testing size, randomly shuffle samples into training set and test set. random_state comparable to see in R
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2019)

X_train

#verifying test and training set rows and column counts
X_train.shape
X_test.shape
y_train.shape
y_test.shape

lreg.fit(X_train, y_train)

#notice that the score on the training set is now 0.617 instead of 0.758
lreg.score(X_test, y_test)

"""
validation set is like a test set within the training set , can think of it as a dress rehearsal / fine tuning set to get ready for the REAL test set

KFold cross validation is a common validation method
the "K" in KFold represents the number of groups for validation


whe K = 5, means we break out training set up into 5 different groups or "folds"

for however many folds you have, you'll also have a corresponding number of "rounds". 

In each round you use one fold for validation and the other folds as training sets, then in the next round the validation fold will rotate 

(e.g. round one uses folds 2-5 for training and fold 1 for validation and folds 2-5 for training, round 2 uses folds 1,3-5 for training and fold 2 for validation, etc )

Each round will have a validation score
"""

from sklearn.model_selection import cross_val_score

#cv is the numerof of folds
scores = cross_val_score(estimator=lreg, X=X_train, y=y_train, cv=10)
scores
np.mean(scores) #sum the scores of each round of validation


#KFold is the more "hands on" version for cross validation
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10) #Initialize KFold

idxs = list(kfold.split(X_train)) #set indices

idxs[0]

val_set = X_train.reset_index(drop=True).iloc[idxs[8][1]] #this is how you would recreate the rows within your training set for your 9th fold, you would change to idxs[7][1] if you wanted to get the 7th fold etc. note: reset_index keeps the shuffled training set in the shuffled order, but resets the index to count them in their new order so that they will correspond to the index numbers in idxs (since KFold basically does the same thing)

y_vals = y_train.reset_index(drop=True).iloc[idxs[8][1]] #

val_set 
y_vals 

coeffs.sort_values(by='Coefficient', ascending=False) #Show coefficients sorted in descending order, helps to find the most impactful coeffiients 

sns.regplot(x='LSTAT', y='PRICE', data=df) #Plot your most important variables against the price. Note that the relationship between these variables and price is not perfectly linear
sns.regplot(x='LSTAT', y='PRICE', order=2, data=df) #When setting order =2 you're seeing the relationship between LSTAT and the relationship between  LSTAT^2 and PRICE
sns.regplot(x='RM', y='PRICE', data=df)
sns.regplot(x='DIS', y='PRICE', data=df)
sns.regplot

X['LSTATSQ'] = X['LSTAT']**2 #Create a new column which is LSTAT Squared based on the regplot w/order=2

"""
1. Re-run train_test_split on new version of X, get X_train, X_test, y_train, y_test

2. input back into cross_val_score function

3. look at scores
"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2019)
scores = cross_val_score(estimator=lreg, X=X_train, y=y_train, cv=10)
scores
np.sum(scores)

#Group search is a way to programmatically go through all of your features and and try raising their order to see what effect that has on the validation score