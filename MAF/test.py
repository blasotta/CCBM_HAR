import torch
from torch.nn import functional as F
#from torchvision.utils import save_image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal

from maf import MAF

from utils.validation import val_maf, val_made
from utils.test import test_maf, test_made
from utils.plot import sample_digits_maf
from datasets.data_loaders import get_data_loaders, get_data

import sys
sys.path.insert(1, 'C:/Users/bened/PythonWork/CCBM_HAR/carrots/eval')
from loaders import load_dataset, load_config, load_conditional_train, load_conditional_val, load_conditional_test

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

dataset = "carrots"
batch_size = 128

# #############Concept works#####################################################
# cn = 15
# string = f"maf_carrots_{cn}_512"
# model = torch.load("model_saves/" + string + ".pt")

# train_dict, lt = load_conditional_train()
# val_dict, lv = load_conditional_val()
# #test_x, test_y, le = load_conditional_test()
# test_dict, lte = load_conditional_test()

# train_x = train_dict[cn].to_numpy()
# val_x = val_dict[cn].to_numpy()
# test_x = test_dict[cn].to_numpy()
# print(test_x.shape)

# train = torch.from_numpy(train_x)
# val = torch.from_numpy(val_x)
# test = torch.from_numpy(test_x)

# train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
# val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
# test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)

# test_maf(model, train, test_loader)
# val_maf(model, train, val_loader)
# #print(lt, lv, lte, sep=' | ')
# #############Concept works#####################################################

train_dict, ltr = load_conditional_train()
#val_x, val_y, le = load_conditional_val()
test_x, test_y, le = load_conditional_test()
val_dict, lt = load_conditional_val()

N = np.shape(test_x)[0]

#array for priors:
priors = []
#dictionary for p(x|y) values, where key=class, value=p(x|key) for all x
lik = {}
val_ll = []

lik_MVN = {}

cc = len(train_dict.keys()) # classs count (i.e. number of distict classes)
for j in range(cc): #calculate p(x|y) and p(y) for all of the 16 classes 0 to 15
    train_x = train_dict[j].to_numpy()
    val_x = val_dict[j].to_numpy()
    
    #### compute mu, cov of multivariate normal for comparison with MAF
    mu = np.mean(train_x, axis=0)
    co = np.cov(train_x, rowvar=0)
    # compute fit of val data to MVN
    ll = multivariate_normal.logpdf(val_x, mean=mu, cov=co)
    M = len(ll)
    print(f'Number of validation examples of class {j} are {M}')
    ll_val = np.sum(ll) / M
    ####
    
    count = np.shape(train_x)[0]
    print(f'The count of class {j} is: ', count)
    prior = count / ltr # calculates prior of class j
    priors.append(prior)
    
    train = torch.from_numpy(train_x)
    val = torch.from_numpy(val_x)
    test = torch.from_numpy(test_x)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)
    
    #load correct model of class j as to calculate p(x|j):
    string = f"maf_carrots_{j}_90"
    model = torch.load("model_saves/" + string + ".pt")
    
    #pdf MVN
    logpdf = multivariate_normal.logpdf(test_x, mean=mu, cov=co)
    lik_MVN[j]=logpdf
    
    test_loss = test_maf(model, train, test_loader)
    #test_loss = test_made(model, test_loader)
    lik[j]=test_loss
    check = val_maf(model, train, val_loader)*-1
    #check = val_made(model, val_loader)*-1
    val_ll.append(check)
    print(f'(MAF) Average validation LL for class {j} is {check}')
    print(f'(MVN) Average validation LL for class {j} is {ll_val}')

log_priors = np.log(priors)
#print('class priors', priors)
#print('log priors: ', log_priors)

result = np.zeros((N,cc))
for i in range(N):
    for c in range(cc):
        log_lik = lik[c][i].numpy() * -1
        result[i,c]= log_lik + log_priors[c]
        
result2 = np.zeros((N,cc))
for i in range(N):
    for c in range(cc):
        log_lik = lik_MVN[c][i]
        result2[i,c]= log_lik + log_priors[c]

y_pred = np.argmax(result, axis = 1)
y_pred2 = np.argmax(result2, axis = 1)
np.savetxt("MAF_results.txt", result, fmt='%.5f', delimiter=" ")
np.savetxt("MVN_results.txt", result2, fmt='%.5f', delimiter=" ")
#print(result)
#print(y_pred2)
#print(test_y)

print('accuracy: ', accuracy_score(test_y, y_pred))
print('accuracy2: ', accuracy_score(test_y, y_pred2))

classes = list(le.classes_)
cf_matrix = confusion_matrix(test_y, y_pred)
print('Number of test samples: ', np.sum(cf_matrix))
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (16,9))
s = sns.heatmap(df_cm, annot=True, cmap="flare", fmt='g')
s.set(xlabel='predicted class', ylabel='true class')
plt.savefig('conf_mat.jpg')

cf_matrix2 = confusion_matrix(test_y, y_pred2)
df_cm2 = pd.DataFrame(cf_matrix2, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (16,9))
s2 = sns.heatmap(df_cm2, annot=True, cmap="flare", fmt='g')
s2.set(xlabel='predicted class', ylabel='true class')
plt.savefig('conf_mat2.jpg')