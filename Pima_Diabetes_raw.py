# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:06:42 2020

@author: Will
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd


data = pd.read_excel(r'C:/Users/Will/Desktop/Projects/Pima_Diabetes.xls')
data1  = np.asarray(data)
Y  = np.asarray(data['Class'])
Pregnant = np.asarray(data['Pregnant'])
Glucose = np.asarray(data['Glucose'])
BP = np.asarray(data['BP'])
Skin = np.asarray(data['Skin'])
Serum = np.asarray(data['Serum'])
BMI = np.asarray(data['BMI'])
Diabetes = np.asarray(data['Diabetes'])
Age = np.asarray(data['Age'])
 

##769 samples
##lets first make some plots that show what were working with.
fig, sub = plt.subplots(2,2 )
plt.subplots_adjust(wspace=.2, hspace=0.4)
titles = ('Age vs. Number of Times Pregnant', 'Age vs. Blood Pressure', 'Age vs. 2-hr Serum Insulin' , 'Age vs BMI')
Ages = (Age,Age,Age,Age)
Att = (Pregnant,BP,Serum,BMI)
for age,stat,title,ax in zip(Ages, Att, titles, sub.flatten()):
    ax.scatter(age,stat,c=Y,cmap = plt.cm.coolwarm, s=5)
    ax.set_title(title)
    
corr = data.corr()
fig, ax = plt.subplots()
im = ax.imshow(corr)

ax.set_title("Pairwise Correlation in Diabetes Data")
ax.set_xticks(np.arange(len(data.columns)))
ax.set_yticks(np.arange(len(data.columns)))
# ... and label them with the respective list entries
ax.set_xticklabels(sorted(data),rotation=45)
ax.set_yticklabels(sorted(data))
ax.figure.colorbar(im,ax=ax)

##in higher dimensions this data will spread out
##first try SVM and report accuracy of different kernels and SVM methods
##divide the dataset in half for training and testing and divide into length 8 vectors for testing
X_train = np.asarray(data.loc[:int(len(data)/2),data.columns!='Class'])
X_train = np.c_[X_train]
##normalize the data
##create a scaler that we can also use for the test data
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

##now make the y training data  
Ytrain = np.asarray(data.loc[:int(len(data)/2),data.columns =='Class'])
Ytrain = np.c_[Ytrain]

##make x and y training data
X_test = data.loc[int(len(data)/2):, data.columns!='Class']
X_test = np.c_[X_test]
X_test_scaled = scaler.transform(X_test)
Ytest = data.loc[int(len(data)/2):, data.columns =='Class']
Ytest = np.c_[Ytest]

##Lets first try SVC with linear kernel
##right now we are are having trouble with the higher degre polynomial..why?
##write a function that plots accuracy of method
def acc_plot(models,titles,label):
    print ("training...")
    model_fit = (clf.fit(X_train_scaled, np.ravel(Ytrain)) for clf in models)
    acc = (clf.score(X_test_scaled,Ytest) for clf in model_fit)
    score = [sc for sc in acc]
    fig, ax = plt.subplots()
    ax.barh([1,2,3,4],score)
    ax.set_xlabel('Fraction Correctly Identified')
    ax.set_title(label)
    ax.set_yticks([1,2,3,4])
    ax.set_yticklabels(titles)
    
    
    
    
C = 1.0
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=1e5),
          svm.SVC(kernel='rbf', gamma=.007, C=26.8),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=1e-4,max_iter=1e8))
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

acc_plot(models,titles,"Accuracy of 4 SVCs")

#3lets do a grid search to find optimal values of c and gamma for rbf kernel:
##from https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
C_range = np.logspace(-2, 10, 8)
gamma_range = np.logspace(-9, 3, 8)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)


print("Searching for optimal parameters...")
grid.fit(X_test_scaled, Ytest)

##wow this is taking a while
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

##now lets plot a heat map of the options:
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,vmin = .65,vmax= .82)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(8), np.logspace(-9,-2,8), rotation=45)
plt.yticks(np.arange(8),np.logspace(-2,5,8))
plt.title('Validation accuracy')
plt.show()

##lets see how logistic regression can do
from sklearn.linear_model import LogisticRegression as LR

log_models = (LR(penalty='l1'),LR(penalty = 'l2'),LR(penalty='elastic net',solver = 'saga'))
log_models = (clf.fit(X_train_scaled, np.ravel(Ytrain)) for clf in log_models)
log_f = (clf.score(X_test_scaled,Ytest) for clf in log_models)
log_score = [sc for sc in log_f]



log_titles = ('L1 Penalty','L2 Penalty','Elastic Net Penalty')

fig, ax = plt.subplots()
ax.barh([1,2,3],log_score)
ax.set_xlabel('Fraction Correctly Identified')
ax.set_title('Accuracy of Three Logistic Regression Models')
ax.set_yticks([1,2,3])
ax.set_yticklabels(log_titles)

##see effect of regularization. See if we can improve each model with better c
##uses stratified k-folds for cross-validation
from sklearn.linear_model import LogisticRegressionCV as LRCV
log_cv = (LRCV(penalty='l1',random_state=0,solver='saga'),LRCV(penalty='l2',random_state=0),LRCV(penalty='elastic net',random_state=0,solver='saga'))
log_cv_models = (clf.fit(X_train_scaled, np.ravel(Ytrain)) for clf in log_cv)
log_cv_f = (clf.score(X_test_scaled,Ytest) for clf in log_cv_models)
log_cv_score = [sc for sc in log_cv_f]
log_titles = ('L1 Penalty','L2 Penalty','Elastic Net Penalty')

model = LRCV().fit(X_train_scaled,np.ravel(Ytrain))
print(np.exp(model.coef_))
print(np.exp(model.intercept_))


fig, ax = plt.subplots()
ax.barh([1,2,3],log_cv_score)
ax.set_xlabel('Fraction Correctly Identified')
ax.set_title('Accuracy of Three Logistic Regression Models (with CV)')
ax.set_yticks([1,2,3])
ax.set_yticklabels(log_titles)

##nope didnt help at all...


##lets try a ridge classifier with built in CV
from sklearn.linear_model import RidgeClassifierCV
clf_ridge = RidgeClassifierCV().fit(X_train_scaled,np.ravel(Ytrain))
ridge_score = clf_ridge.score(X_test_scaled,Ytest)
print('The Ridge Classifier correctly predicted %s percent of test cases.' %(ridge_score*100))

##lets try a Neural Net - multlayer perceptron
from sklearn.neural_network import MLPClassifier as MLP
MLP_models = (MLP(activation = 'logistic').fit(X_train_scaled,np.ravel(Ytrain)),
                  MLP(activation = 'tanh').fit(X_train_scaled,np.ravel(Ytrain)),
                MLP(activation = 'relu').fit(X_train_scaled,np.ravel(Ytrain) ))
MLP_f = (clf.score(X_test_scaled,Ytest) for clf in MLP_models)
MLP_score = [sc for sc in MLP_models]

log_titles = ('Logistic Activation','Hyperbolic Tangent Activation','Rectified Linear Activation')

fig, ax = plt.subplots()
ax.barh([1,2,3],log_cv_score)
ax.set_xlabel('Fraction Correctly Identified')
ax.set_title('Accuracy of Multi-Layer Perceptron Models')
ax.set_yticks([1,2,3])
ax.set_yticklabels(log_titles)