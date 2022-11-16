import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(1, 'C:/Users/bened/PythonWork/CCBM_HAR/carrots/eval')
from loaders import one_hot_encode, get_carrots, load_conditional_test, get_UCIHAR

num_cond_inputs = 16

# Loading and preprocessing data CARROTS
trn_x, trn_y, v_x, v_y, log_priors = get_carrots()
tst_x, tst_y, le = load_conditional_test()
troh_y = one_hot_encode(trn_y, num_cond_inputs)
vaoh_y = one_hot_encode(v_y, num_cond_inputs)
teoh_y = one_hot_encode(tst_y, num_cond_inputs)

# For UCI HAR
# trn_x, trn_y, v_x, v_y, log_priors, tst_x, tst_y, classes = get_UCIHAR()
# trn_y = trn_y.astype('int')
# v_y = v_y.astype('int')
# tst_y = tst_y.astype('int')
# troh_y = one_hot_encode(trn_y, num_cond_inputs)
# vaoh_y = one_hot_encode(v_y, num_cond_inputs)
# teoh_y = one_hot_encode(tst_y, num_cond_inputs)

# No need to change this
train_x = trn_x.astype('float32')
train_y = troh_y.astype('int')
val_x = v_x.astype('float32')
val_y = vaoh_y.astype('int')
test_x = tst_x.astype('float32')
test_y = teoh_y.astype('int')

## Aproach 1, USING p(x,y)

#Combining data and labels to learn joint density p(x,y)
join_trn = np.c_[train_x, train_y]
join_val = np.c_[val_x, val_y]


#### compute mu, cov of multivariate normal for comparison with MAF
mu = np.mean(join_trn, axis=0)
co = np.cov(join_trn, rowvar=0)
# compute fit of val data to MVN


lik = {}
N = test_x.shape[0]

for i in range(num_cond_inputs):
    # iterate through the classes to calculate p(x|y)
    # first one hot encode class to use for prediction
    a = np.full(N, i, dtype=int)
    pred_y = one_hot_encode(a, num_cond_inputs)
    
    join_tst = np.c_[test_x, pred_y]

    log_pxy = multivariate_normal.logpdf(join_tst, mean=mu, cov=co, allow_singular=True)

    lik[i] = log_pxy

result = np.zeros((N, num_cond_inputs))
for i in range(N):
    for c in range(num_cond_inputs):
        log_lik = lik[c][i]
        result[i, c] = log_lik + 0*log_priors[c]

y_pred = np.argmax(result, axis=1)

print('Joint MVN accuracy: ', accuracy_score(tst_y, y_pred))

## Aproach 2, USING p(x|y)

# Let's try learning one MVN per class instead of the joint distribution so
# p(x|y)
D = np.c_[trn_x, trn_y]

df = pd.DataFrame(D)
# FOR CARROTS
df.rename(columns={30: "class"}, inplace = True)
# FOR UCIHAR
# df.rename(columns={561: "class"}, inplace = True)

data = df.astype({'class': 'int32'})

#group by class for conditional density estimation.
datasets = {} #dictionary with key=class and value=data pairs
by_class = df.groupby('class')

for groups, data in by_class:
    d = data.drop(columns='class')
    datasets[groups] = d
    
    
lik_MVN = {}
for j in range(num_cond_inputs):
    train_x = datasets[j].to_numpy()
    #### compute mu, cov of multivariate normal for comparison with MAF
    mu = np.mean(train_x, axis=0)
    co = np.cov(train_x, rowvar=0)
    
    
    logpdf = multivariate_normal.logpdf(test_x, mean=mu, cov=co, allow_singular=True)
    lik_MVN[j]=logpdf
    
result2 = np.zeros((N,num_cond_inputs))
for i in range(N):
    for c in range(num_cond_inputs):
        log_lik = lik_MVN[c][i]
        result2[i,c]= log_lik + log_priors[c]
        
np.savetxt("MVN_Benchmark_results.txt", result2, fmt='%.5f', delimiter=" ")
y_pred2 = np.argmax(result2, axis = 1)
print('Split MVN accuracy: ', accuracy_score(tst_y, y_pred2))

classes = list(le.classes_)
cf_matrix = confusion_matrix(tst_y, y_pred2)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (16,9))
s = sns.heatmap(df_cm, annot=True, cmap="flare", fmt='g')
s.set(xlabel='predicted class', ylabel='true class')
plt.savefig('Conf_MVN_Benchmark.jpg')