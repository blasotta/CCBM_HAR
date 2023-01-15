import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, accuracy_score
from scipy.special import logsumexp
from dataloader.loaders import get_carrots, load_conditional_test, get_UCIHAR
from dataloader.preprocess_motion_sense import get_moSense


def mvn_loader(dataset_name, cond_inputs, window, win_size, trn_step, augment, noise):
    print('--------Loading and Processing Data--------')
    if dataset_name == 'CARROTS':
        trn_x, trn_y, v_x, v_y, log_priors = get_carrots(window, win_size, trn_step, augment, noise)
        tst_x, tst_y, le = load_conditional_test(window, win_size, win_size)
    elif dataset_name == 'UCIHAR':
        trn_x, trn_y, v_x, v_y, log_priors, tst_x, tst_y, le = get_UCIHAR()
        trn_y = trn_y.astype('int')
        v_y = v_y.astype('int')
        tst_y = tst_y.astype('int')
    elif dataset_name == 'MOSENSE':
        trn_x, trn_y, v_x, v_y, log_priors, tst_x, tst_y, le = get_moSense(window, win_size, trn_step, augment, noise)
        trn_y = trn_y.astype('int')
        v_y = v_y.astype('int')
        tst_y = tst_y.astype('int')
    else:
        print('Only the datasets CARROTS and UCIHAR are supported currently')
        
    trn_x = trn_x.astype('float32')
    tst_x = tst_x.astype('float32')
    N = tst_x.shape[0]
        
    return trn_x, trn_y, tst_x, tst_y, N, log_priors, le, cond_inputs

def get_transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = np.zeros((n,n))

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    M += 1
    #now convert to probabilities:
    M = M/M.sum(axis=1, keepdims=True)
    return M

def log_matmul(A,B):
    Astack = np.stack([A]*B.shape[1]).transpose(1,0,2)
    Bstack = np.stack([B]*A.shape[0]).transpose(0,2,1)
    return logsumexp(Astack+Bstack, axis=2)


def mvn_bench(trn_x, trn_y, tst_x, tst_y, N, log_priors, num_cond_inputs):
    # Let's try learning one MVN per class so p(x|y)
    D = np.c_[trn_x, trn_y]
    idx = D.shape[1]-1
    df = pd.DataFrame(D)
    df.rename(columns={idx: "class"}, inplace = True)

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
        
        
        logpdf = multivariate_normal.logpdf(tst_x, mean=mu, cov=co, allow_singular=True)
        lik_MVN[j]=logpdf
        
    '''
    Bayes classifier
    LL is the classes x N array containing log p(x|y=c) for all classes c in C over all samples
    First calculate numerator of Bayes' as likelihood + prior in log space for all samples and classes
    Then get prediction as argmax over each column
    '''
    LL = np.array(list(lik_MVN.values()))
    result = LL + log_priors.reshape(-1,1)
    y_pred = np.argmax(result, axis=0)
    
    
    '''
    HMM recognition model
    LL is the classes x N array containing log p(x|y=c) for all classes c in C over all samples
    First calculate prediction step of HMM by matmul of prior and transition matrix A 
    Then get correction by updating with observation, prediction is argmax over states
    '''
    trajectory = []
    pi = log_priors.reshape(1,-1)
    A = np.log(get_transition_matrix(trn_y))
    for t in range(N):
        log_pred = log_matmul(pi, A)
        log_num = log_pred + LL[:,t]
        log_res = log_num - logsumexp(log_num)
        trajectory.append(np.argmax(log_res))
        pi = log_res
        
    s = np.array(trajectory)
            
    # np.savetxt("MVN_pxy.txt", LL, fmt='%.5f', delimiter=" ")
    # y_pred2 = np.argmax(result2, axis = 1)
    ll = np.sum(np.amax(LL, axis=0))/N
    
    bay_acc = accuracy_score(tst_y, y_pred)
    hmm_acc = accuracy_score(tst_y, s)
    
    return bay_acc, hmm_acc, ll

def run_mvn(dataset_name, cond_inputs, window, win_size, trn_step, augment, noise):
    trn_x, trn_y, tst_x, tst_y, N, log_priors, le, num_cond_inputs = mvn_loader(dataset_name, cond_inputs, window, win_size, trn_step, augment, noise)
    bay_acc, hmm_acc, ll = mvn_bench(trn_x, trn_y, tst_x, tst_y, N, log_priors, num_cond_inputs)
    return bay_acc, hmm_acc, ll