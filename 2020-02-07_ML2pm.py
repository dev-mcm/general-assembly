import pandas as pd
import numpy as np



"""
Note: FYI: KFolds and time
"""

#FYI Kfolds is a bad way to test when you are trying to predict the future, when your test set is probably going to be your most recent set of data, and in that case your test set should probably be the most recent data set, and your validation set should be the most recent data before that. Example: predicting General Assembly attendence next semester

#HOWEVER if time doesn't have a huge impact on the outcome (like if your data is fairly consistent across time), then you could still use KFolds

#When you have sequential data like this, you might not want to shuffle it, you could test for this by using KFolds 
#Using the General Assembly Attendance Example, it might be that only the past 5 years of General Assembly attendance data is relevant for predicint next semester's attendance. If you use KFolds and keep the data in order, you can look for a breaking point where data becomes much more or much less accurate after a certain point, and then you could limit your data set to data that occurred after that breaking point

""" 
Working with Dates in Pandas
"""

# bit.ly/date-offsets 
# df['date_column'].astype(np.int64) #this will change the date to unixtime , might want to divide by 100000000000 to make teh value less enormous since unixtime is in.... miliseconds? 

"""
CLASSIFICATION
with Decision Trees
"""


#DIFFERENT DATASET, DIFFERENT TYPE OF PROBLEM
#Going to try to predict who made it off the titantic and who did not
df = pd.read_csv('data/titanic.csv')

df.head() #Check new dataset
#Pclass = class, 1 is first class, 2 is business class, 3 is Jack
#SibSp = number of siblings
#Parch = whether they were parents with children
#Cabin = cabin number
#Embarked = what port they were picked up from

"""
Note: You can do classification with regression through Logistic regression. Decision trees can also be used for linear models

F1Score - rate of false positives and false negatives vs true positives and true negatives 
you can also focus more on false positives or false negatives (going to talk about this more later)

Nice thing about decision trees is that it is not affected by the scale of your data
"""

from sklearn.tree import DecisionTreeClassifier #FYI there's also DecisionTreeRegressor, difference is that it takes the average value instead of the prediction
#from prep import_drawtree #Jonathan can send us the program to support this after teh class

tree = DecisionTreeClassifier()

X = pd.get_dummies(df[['Sex', 'Fare','Pclass','SibSp']])
y = df['Survived']
tree.fit(X,y)

tree.predict(X) #Predicts for each person whether they lived or died 
tree.predict_proba(X)

X.head()

"""
The above generates a single decision tree which tends to overfit and are a little erratic and on their own are not a great predictor for out of sample (test) data. BUT a random forest takes a set of decision trees and then takes the average of those results which ends up being pretty useful
"""

#If we changed y to be the embarked column

X = pd.get_dummies(df[['Sex', 'Fare','Pclass','SibSp']])
y = df['Embarked']
y = y.fillna('S') #have to fill empty values
tree.fit(X,y)

tree.predict(X) #Predicts for each person whether they lived or died 
tree.predict_proba(X) #notice there are now three columns, one for each category in df['Embarked']

y = df['Survived'] #reset y to df['Survived']



"""
RANDOM FOREST
Ensamble of Decision Trees 
1. Grow multiple Trees
2. Each tree comes from 2/3 of your dataset, randomly sampled from your training set
3. Random forest will also randomly sample the columns from your dataset to consider at each node / branch point  
"""
from sklearn.ensemble import RandomForestClassifier  # also has RandomForestRegressor

rfc = RandomForestClassifier()

rfc.get_params()
#max depth sets the depth of the tree, defaults to 
#max_features sets the % of columns to randomly sample, defaults to "auto" but 0.2 would sample 20% of variables
#min_samples_leaf defaults to 1 sample to leaf, but you can make this number larger to prune the tree
#n_estimators defaults to 100, sets the number of trees you're growing in the forest

df.head() #Notice that our dataset has text! this is where pd.get_dummies() comes in

"""
Text based data, can be ordinal (has a natural order to it; e.g. small, medium, large, etc

S = 0
M = 1
L = 2
XL = 3

you can think of Pclass as a categorical variable

Nominal: has no inherent order, these are what you use dummy variables for 

"""

pd.get_dummies(df['Embarked']) #Notice that it creates a new yes/no column for each possible category

df['Greeting'] = df['Name'].str.split().str[1] #splits the name by default it splits on space, you can pass ',' or something in the arguments to split on something else 

df['Name'].nunique() #Can see that there are 891 different names
df['Greeting'].nunique() # see number of different unique greetings

df.head()

df['Ticket'].str.split().str[1]

X = df[['Greeting','Sex','Embarked','Fare','SibSp','Parch']]
y = df['Survived']

X = pd.get_dummies(X) #create binary columns for nominal categories 

from sklearn.model_selection import GridSearchCV

rfc #take alook at estimators again 

#Now let's build a dictionary with the different versions of the parameters we're going to test
params = {
    'min_samples_leaf': [1,5,10,15,25],
    'max_features': [0.4, 0.5, 0.6, 0.7],
    'n_estimators': [10, 25, 50]
}

rfc.get_params() #class_weight tells you how much attention you want to pay to your different labels

rfc.set_params(class_weight={0: .25, 1: 0.75 }) #this would make sure that we accurate predict people who survive (where prediction is 1) and mean that it cares less about false negatives 




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2019, stratify=y) #Stratify makes sure that you have equal representation of your different labels in your trainin and test sets (aka balances the number of survivors )


from sklearn.model_selection import StratifiedKFold 
kfold = StratifiedKFold(n_splits=10) #makes sure there's equal distribution of Y in each kfold

#by default accuracy 

from sklearn.metrics import balanced_accuracy_score #can use this to make sure it considers importance on NOT missing survivors, We have to do more to this before we can add the parameter grid = GridSearchCV(estimator=rfc, param_grid = params, cv=kfold) #set cv to kfolds to use the stratified kfolds, otherwise you could just set it to 10

grid = GridSearchCV(estimator=rfc,  param_grid = params, cv=kfold) #set cv to kfolds to use the stratified kfolds, otherwise you could just set it to 10

grid.fit(X_train, y_train)

grid.best_params_ 

grid_results = pd.DataFrame(grid.cv_results_)
grid_results #lets you see the diagnostics for 


"""
Feature Selection
through feature importance

Lasso is useful for figuring out what columns to use

Or Logistic Regression for classification
"""

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.get_params() #see where it says penalty, penalty is something that shrinks your weights for you. Penalities come in two different flavors, L1 and L2. Useful when you have a larger number of low quality variables 

#L1 shrinks non-impactful variables to zero, helps you figure out which columns to exclude 
#C: determines how quickly it shirnks values to zero, the smaller C is, the stronger the penalty becomes. 

logreg.set_params(penalty='L1') #not everything works with the L1 penalty, have to do the below solver and max_iter params to get it to work 

logreg.penalty='l1'
X


logreg.solver = 'saga'


logreg.max_iter = 1000

logreg.fit(X,y)

logreg.coef_ #Look at the variables that were set to 0, and those can probably be removed from your random forest model

rfc = RandomForestClassifier()


"""
Neural networks are used for unstructured data
- Computationally the most expensive
- Requires a GPU to run
- Dificult to train, Expensive to train

Unique powerful for finding subtle, non-linear patterns inside unstructured data 

In Linear regression, you multiple X by the coefficient (or weight) in order to get Y
Neural networks add an additional layer of abstraction before getting to Y, can think of it as a computational slow cooker

X * weight + X_hidden*weight + X_Hidden2*weight etc = Y

"Deep Learning" refers to a neural network with 2 hidden layers

building neural networks i basically determining how you want to connect the different layers

The final layer should always have a layer of weights = to the number of categories you're trying to predict (so in the bitly example there are 10 categoreis and 10 weights)
"""

#bit.ly/image_classifier

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.fashion_mnist

(train_img, train_label), (test_img, test_label) = mnist.load_data()

train_img.shape

train_img[0]

train_label

plt.figure()
plt.imshow(train_img[666])

train_img[0].flatten()

train_img = train_img / 255.0

test_img = test_img / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='sgd', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_img, train_label, epochs=10)

model.get_weights()[2]

train_label[0]

test_loss, test_acc = model.evaluate(test_img, test_label)

""" 
There are three types of Neural Network layers taht tend to get used

Dense : linear combinations
keras.layers.Dense
- forms a connection between each  value and every other  value
Convolutional: Images
- Convolutional layer will form a connection between each pixel and each surrounding pixel, think of it as like a box that moves around each pixel and it forms a connection to all of the other values within the box
Recurring: sequence models: Test, audio
- if the value of one thing is really dependent on whatever comes before or after
- kind of similar to a for loop where it will multiple a value by its weight a certain number of times 
- there are a few different types of recurrent layers you can user
--LSTM Long S? Term Memory -- most highly powered recurrant network, good at calculating something across multiple steps, more computationally expensive. Good for something like learning blocks of text in 50 word increments 
--GRU Gated Recurring Unit -- faster at training
"""
