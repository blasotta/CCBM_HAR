# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:14:31 2022

@author: bened
"""
from sklearn import model_selection
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from loaders import load_dataset, load_config
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


config = load_config('qda_config.yaml')

X_1, y_1,_ = load_dataset(1)
X_2, y_2,_ = load_dataset(2)
X_3, y_3,_ = load_dataset(3)
X_4, y_4,_ = load_dataset(4)
X_5, y_5,_ = load_dataset(5)
X_6, y_6,_ = load_dataset(6)
X_7, y_7,le = load_dataset(7)

#Test
#A bit of an arbitrary choice, but select subject 3 as final test set (original CCBM shows worst perfromance on this set)
#and use remaining subjects data for training and evaluation for model selection. Cross validation will be performed on the
#remaining sets.
#X = [X_1, X_2, X_4, X_5, X_6, X_7]
#y = [y_1, y_2, y_4, y_5, y_6, y_7]

#X_test = X_3
#y_test = y_3

#Using subject 7 as test and rest for training
X = [X_1, X_2, X_3, X_4, X_5, X_6]
y = [y_1, y_2, y_3, y_4, y_5, y_6]

X_test = X_7
y_test = y_7

#TODO: Probably add second loop such that always one subject is tested on and report avg. Performance across all subject
#datasets in the end.

X_train = np.concatenate(X, axis=0)
y_train = np.concatenate(y, axis=0)

#List containing all models that should be tested, append new models here
#models are a tuple of a string identifier and the corresponding model.
models = []
models.append(('QDA', QDA(priors=config['priors'], reg_param=config['reg_param'])))
models.append(('LDA', LDA()))

#Set up k-fold cross validation. Since the dataset contains data from 6 subjects k=6 seems like a natural choice
n_folds = config['folds']
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=n_folds)
    print("Testing model: ", name)
    cv_results = model_selection.cross_val_score(
        model,
        X_train,
        y_train,
        cv=kfold,
        scoring=config['scoring'], #f1_micro score; micro is preferrable for imbalanced classes over macro, same as 'accuracy' for multi class but single label classification
        verbose=config['verbose'], 
        n_jobs=config['n_jobs'])
    results.append(cv_results)
    names.append(name)
    msg = f"{name}, Mean acc: {cv_results.mean()}, with std: {cv_results.std()}"
    print(msg + "\n")
    
fig = plt.figure(figsize=(12,7))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel(config['scoring'])
plt.show()
plt.savefig('box_plot.jpg')

#Use best model, fit on entire training data and show confusion matrix
classes = list(le.classes_)
model = QDA()
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(f'Using QDA an Prediction accuracy of {acc} was achieved.')
y_pred = model.predict(X_test)

print('f1_score (micro): ', f1_score(y_test, y_pred, average='weighted')) # include macro

cf_matrix = confusion_matrix(y_test, y_pred)
print('Number of test samples: ', np.sum(cf_matrix))
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (16,9))
s = sn.heatmap(df_cm, annot=True, cmap="flare", fmt='g')
s.set(xlabel='predicted class', ylabel='true class')
plt.savefig('conf_mat.jpg')