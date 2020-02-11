#python test script

import numpy as np

var  = np.arange(20).reshape(5,4)

var.mean()

#take the first three rows of columns in index position 2 and 3
var[:3, 2:4]

import pandas as pd

df = pd.read_csv('data/housing.csv')

df.head()

#return a number of descriptive statistics for each  numerical column (count, mean, std, min, etc)
df.describe()

#Return mean statistic for every numerical column
df.mean()

#find the number of empty / null values in each column
df.isnull().sum()

#
df[['TAX', 'NOX', 'CRIM']].describe()

#grab the first 100 rows of these three columns, and calculate their mean
df[['TAX', 'NOX', 'CRIM']][:100].mean()

#grab the first 50 rows of columns in index position 3 through 12 (not including column w/index 12)
df.iloc[:50, 3:12]

# grab the first 50 rows of columns in columns w/index position 2,3, and 0
df.iloc[:50, [2,3,0]]

#what is the average of the first 100 rows of the CHAS column?
df.iloc[:50,3].mean()

#Using their labels, what is the average value of the first 100 rows of the 'B' and 'CHAS' columns?
df[['B','CHAS']][:100].mean()

#What are the median values of the first 50 rows of the columns at index position 0 and 2?
df.iloc[:50,[0,2]].median()

#find rows w/ taxes > 350
df['TAX'] >  350

#return rows w/taxes > 350
df[df['TAX'] > 350][['CHAS','NOX']].max()

# multiple conditions
df[(df['TAX'] > 350) & (df['CHAS'] ==0)]

# FYI the "CHAS" column encodes whether a house was on the lake or not, 1 = on the lake, = not on the lake

df[(df['TAX'] > 350) & (df['CHAS'] ==0)].shape

#What is the average value of the PRICE column when taxes are above 300?
df[df['TAX'] > 300]['PRICE'].mean() 

#What is the average value of the PRICE column when taxes are above 300 and CHAS = 1
df[(df['TAX'] > 300) & (df['CHAS'] == 1)]['PRICE'].mean() 

#How many homes are both on the lake, with crime rates that are above average
df[(df['CHAS'] == 1) & (df['CRIM'] > df['CRIM'].mean())].shape[0]


#Useful methods
df.describe() #returns summary stats for each numerical column
df.info() #returns data frame schema 
df.isnull() #returns null values
df.isnull().sum() #count number of null values in each column
df['TAX'].nlargest(5) #return 5 largest values in the 'TAX' column
df.value_counts()  #### this gives the count for each value in a series (doesn't work on a dataframe)
df['CHAS'].value_counts() 
df.count() #this works though
df.sort_values([by], ascending=[bool])
#good rsource for how to do stuff w/pandas https://pandas.pydata.org/docs/user_guide/index.html

import matplotlib as mpl
import seaborn as sns

df['PRICE'].plot(kind='hist')

sns.regplot(x='LSTAT', y='PRICE', order =2, data =df)


#Three definitions of machine learning
"""
1. The borader notion of building statistical artifacts that become more accurate over time based on experience 
2. Linear Algebra + Statistical Analysis, written in code
3. cost = sum(answer - guess)^2


Different types of Machine Learning

Regression (Linear) vs. Classification (Categorical)
Supervised (Training data) vs. Unsupervised (No verification, e.g. Clustering)
Structured (normal w/something like dataframe) vs. Unstructured (images and long text, no labels)

Image classifier: Unstructured, Classification, Supervised
Cluster Analysis/segmentation: Unsupervised
Dimensionality Reduction: when you want to shrink the size of a large dataset while still maintaining the same quality of core information that you need -- common example of unsupervised learning

GLM: Generalized Linear Model - Structured, supervised regression
Ensemble: Structured, Supervised, Classification
Neural Network: Unstructured, Unsupervised, Classification
"""

#Scikit Learn
"""
main library used to implement machine learning
Jack of all trades, master of none
contains built-in techniques for most ML concepts
is primary built to access your own computer's memory
Runs on a CPU, but not a GPU (Which means it doesn't work as well for machine learning)
"""

fit() # apply algorithm to you r data
score() #evaluate algorithm


from sklearn.linear_model import LinearRegression

X = df[['LSTAT','TAX']]
y = df[['PRICE']]

#step you have to take to initialize what you import from scikit leran
lreg = LinearRegression()

lreg.fit(X,y)

lreg.coef_ #returns the two slopes for x1 and x2
lreg.intercept_ #returns the intercept 

X.head(5)

lreg.predict(X.head(1))

X.head(1)*lreg.coef_ #returns list of X1 and X2 multiplied by their coefficients
np.sum(X.head(1)*lreg.coef_, axis=1) #sums the two values  returned by X.head(1)*lreg.coef_ #setting aXis =1 means it will sum across the columns instead of down the columns
np.sum(X.head(1)*lreg.coef_, axis=1) + lreg.intercept_ #adds the intercept to the above

#score of the model
lreg.score(X,y)


#re-run the linear regression with every column except PRICE
X = df.iloc[:,:-1]

lreg.fit(X,y) 
lreg.coef_
lreg.intercept_
lreg.score(X,y)


X.columns


coeff_dict = {
    'Variable': X.columns,
    'Coefficient': lreg.coef_[0] # had to add [0] because apparently the version of numpy or scikitlearn turns lreg.coef_ into an array instead of a list 
}

coeffs = pd.DataFrame(coeff_dict)
coeffs
# The size of the coefficient KIND OF relates to importance, but it's also affected by the range of the input variables. For example NOX has a coefficient of -17, it only ranges from 0.385 to 0.871

np.mean(X - X.mean())

#Demeaning the data, devided by the standard deviation
X = (X - X.mean()) / X.std()

X.describe()

lreg.fit(X,y)

lreg.coef_ #returns the two slopes for x1 and x2
lreg.intercept_ #returns the intercept 
coeff_dict = {
    'Variable': X.columns,
    'Coefficient': lreg.coef_[0] # had to add [0] because apparently the version of numpy or scikitlearn turns lreg.coef_ into an array instead of a list 
}
coeffs = pd.DataFrame(coeff_dict)
coeffs.



df['TAX'].plot(kind='hist')

X['TAX'].plot(kind='hist')

X.mean()

df['RM'].describe()

#NEXT TIME: Cross validation, More modern version of linear regression (glm, better suited for large data sets)
#New data set for classification, random forest & Deep learning w/neural networks
