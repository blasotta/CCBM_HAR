# import utils
import flows as fnn
# import datasets
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.special import logsumexp
from MVN_Benchmark import run_mvn
import pandas as pd

import sys
sys.path.insert(1, 'C:/Users/bened/PythonWork/CCBM_HAR/carrots/eval')
from loaders import one_hot_encode, get_carrots, load_conditional_test, get_UCIHAR
from preprocess_motion_sense import get_moSense

if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')

flow = "maf"  # 'maf' or 'made'
# dataset_name = "CARROTS"  # can be 'mnist', 'power', 'hepmass'
batch_size = 128 # 128 seems good
test_bs = 128 # can be 128
num_blocks = 5  # number of mades in maf, default 5
num_hidden = 512  # 1024 or 512 for mnist, 100 for power, 512 for hepmass
lr = 5e-5  # 5 works well
weight_decay = 1e-6 # 1e-3 # default 6
patience = 5  # For early stopping
seed = 42
max_epochs = 2  # 1000
cond = True
no_cuda = False
# num_cond_inputs = 16 # 16 for carrots, 6 for UCIHAR, None if cond = False

cuda = not no_cuda and torch.cuda.is_available()
print('CUDA:', cuda)
device = torch.device("cuda:0" if cuda else "cpu")
print('Device:', device)

torch.manual_seed(seed)

def load_data(dataset_name, num_cond_inputs, window, win_size, trn_step, augment, noise):
    print('--------Loading and Processing Data--------')
    if dataset_name == 'CARROTS':
        trn_x, trn_y, v_x, v_y, log_priors = get_carrots(window, win_size, trn_step, augment, noise)
        tst_x, tst_y, le = load_conditional_test(window, win_size, win_size)
        troh_y = one_hot_encode(trn_y, num_cond_inputs)
        vaoh_y = one_hot_encode(v_y, num_cond_inputs)
        teoh_y = one_hot_encode(tst_y, num_cond_inputs)
    elif dataset_name == 'UCIHAR':
        trn_x, trn_y, v_x, v_y, log_priors, tst_x, tst_y, le = get_UCIHAR()
        trn_y = trn_y.astype('int')
        v_y = v_y.astype('int')
        tst_y = tst_y.astype('int')
        troh_y = one_hot_encode(trn_y, num_cond_inputs)
        vaoh_y = one_hot_encode(v_y, num_cond_inputs)
        teoh_y = one_hot_encode(tst_y, num_cond_inputs)
    elif dataset_name == 'MOSENSE':
        trn_x, trn_y, v_x, v_y, log_priors, tst_x, tst_y, le = get_moSense(window, win_size, trn_step, augment, noise)
        trn_y = trn_y.astype('int')
        v_y = v_y.astype('int')
        tst_y = tst_y.astype('int')
        troh_y = one_hot_encode(trn_y, num_cond_inputs)
        vaoh_y = one_hot_encode(v_y, num_cond_inputs)
        teoh_y = one_hot_encode(tst_y, num_cond_inputs)
    else:
        print('Only the datasets CARROTS, UCIHAR and MOSENSE are supported currently')
        
    print('--------Correcting Data Type--------')
    train_x = trn_x.astype('float32')
    train_y = troh_y.astype('int')
    val_x = v_x.astype('float32')
    val_y = vaoh_y.astype('int')
    test_x = tst_x.astype('float32')
    test_y = teoh_y.astype('int')
    
    D = train_x.shape[1]
    
    print('Building Data Tensors')
    train_tensor = torch.from_numpy(train_x)
    train_labels = torch.from_numpy(train_y)
    train_dataset = torch.utils.data.TensorDataset(
        train_tensor, train_labels)

    valid_tensor = torch.from_numpy(val_x)
    valid_labels = torch.from_numpy(val_y)
    valid_dataset = torch.utils.data.TensorDataset(
        valid_tensor, valid_labels)

    test_tensor = torch.from_numpy(test_x)
    test_labels = torch.from_numpy(test_y)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)

    print('Creating Data Loaders')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=test_bs, shuffle=False, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=False, drop_last=False)
    
    return train_loader, valid_loader, test_loader, test_tensor, tst_y, D, log_priors, le, trn_y


def initialize_model(flow, num_inputs, num_cond_inputs):
    print('Initializing Model')
    print('Number of features:', num_inputs)
    act = 'relu'

    modules = []

    assert flow in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
    if flow == 'glow':
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        print("Warning: Results for GLOW are not as good as for MAF yet.")
        for _ in range(num_blocks):
            modules += [
                fnn.BatchNormFlow(num_inputs),
                fnn.LUInvertibleMM(num_inputs),
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu')
            ]
            mask = 1 - mask
    elif flow == 'realnvp':
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        for _ in range(num_blocks):
            modules += [
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs)
            ]
            mask = 1 - mask
    elif flow == 'maf':
        for _ in range(num_blocks):
            modules += [
                fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
            ]
    elif flow == 'maf-split':
        for _ in range(num_blocks):
            modules += [
                fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                              s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
            ]
    elif flow == 'maf-split-glow':
        for _ in range(num_blocks):
            modules += [
                fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                              s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs),
                fnn.InvertibleMM(num_inputs)
            ]

    model = fnn.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)
                
    return model


def train(epoch, model, train_loader, optimizer):
    # global global_step, writer
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        # Gradient Norm clipping Test
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0)
        # Test
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))

        # writer.add_scalar('training/loss', loss.item(), global_step)
        # global_step += 1

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if cond:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device),
                  train_loader.dataset.tensors[1].to(data.device).float())
    else:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1
            
    print('Epoch Complete')


def validate(epoch, model, loader, prefix='Validation'):
    print(f'Beginning validation epoch {epoch}')
    # global global_step, writer

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            # sum up batch loss
            # val_loss += -model.log_probs(data, cond_data).sum().item() # ORIGINAL LINE
            batch_log_probs = model.log_probs(data, cond_data)
            corrected = torch.nan_to_num(batch_log_probs, nan=-100, posinf=1000, neginf=-100)
            #detect nan or inf values:
            nan = torch.isnan(batch_log_probs).sum().item()
            inf = torch.isinf(batch_log_probs).sum().item()
            if (nan >= 1) or (inf >= 1):
                print(f'{nan} log densities in batch {batch_idx} are nan {inf} are inf')
            # if batch_idx == 212:
            #     print(batch_log_probs)
            sum_log_prob = -corrected.sum().item()
            val_loss += sum_log_prob
            ##### In this sum the value inf appears some time find out why
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    # writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

    pbar.close()
    return val_loss / len(loader.dataset)


def get_log_pxy(epoch, model, loader):

    log_pxy = []
    model.eval()
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            res = torch.flatten(model.log_probs(data, cond_data))
            a = res.tolist()
            log_pxy.extend(a)

    return log_pxy

def get_transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def evaluate(data, labels, trn_y, epoch, model, log_priors, num_cond_inputs):
    # run Bayes classifier prediction
    lik = {}
    N = labels.shape[0]

    for i in range(num_cond_inputs):
        # iterate through the classes to calculate p(x|y)
        # first one hot encode class to use for prediction
        a = np.full(N, i, dtype=int)
        pred_y = one_hot_encode(a, num_cond_inputs)
        pred_labels = torch.from_numpy(pred_y)
        pred_dataset = torch.utils.data.TensorDataset(data, pred_labels)

        pred_loader = torch.utils.data.DataLoader(
            pred_dataset,
            batch_size=test_bs,
            shuffle=False,
            drop_last=False)

        log_pxy = get_log_pxy(epoch, model, pred_loader)

        lik[i] = log_pxy

    '''
    Bayes classifier
    LL is the classes x N array containing log p(x|y=c) for all classes c in C over all samples
    First calculate numerator of Bayes' as likelihood + prior in log space for all samples and classes
    Then get prediction as argmax over each column
    '''
    LL = np.array(list(lik.values()))
    result = LL + log_priors.reshape(-1,1)
    y_pred = np.argmax(result, axis=0)
    
   
    # np.savetxt("MAF_pxy.txt", LL, fmt='%.2f', delimiter=" ")
    
    '''
    HMM recognition model
    LL is the classes x N array containing log p(x|y=c) for all classes c in C over all samples
    First calculate prediction step of HMM by matmul of prior and transition matrix A 
    Then get correction by updating with observation, prediction is argmax over states
    '''
    trajectory = []
    pi = np.exp(log_priors)
    A = get_transition_matrix(trn_y)
    for t in range(N):
        log_pred = np.log(pi.T@A)
        log_num = log_pred + LL[:,t]
        log_res = log_num - logsumexp(log_num)
        trajectory.append(np.argmax(log_res))
        pi = np.exp(log_res)
        
    s = np.array(trajectory)

    bay_accuracy = accuracy_score(labels, y_pred)
    # bay_f1_mac = f1_score(labels, y_pred, average='macro')
    # bay_f1_wei = f1_score(labels, y_pred, average='weighted')
    hmm_accuracy = accuracy_score(labels, s)
    return bay_accuracy, hmm_accuracy, y_pred


def plot_confMat(le, tst_y, y_pred):
    try:
        classes = list(le.classes_)
    except:
        classes = le
    
    cf_matrix = confusion_matrix(tst_y, y_pred)
    print('Number of test samples: ', np.sum(cf_matrix))
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                          columns = [i for i in classes])
    plt.figure(figsize = (16,9))
    s = sns.heatmap(df_cm, annot=True, cmap="flare", fmt='g')
    s.set(xlabel='predicted class', ylabel='true class')
    plt.savefig('conf_mat.jpg')


def run(dataset, cond_inputs, window, win_size, trn_step, augment, noise, plot=False):
    train_loader, valid_loader, test_loader, test_tensor, tst_y, D, log_priors, le, trn_y = load_data(dataset, cond_inputs, window, win_size, trn_step, augment, noise)
    model = initialize_model('maf', D, cond_inputs)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('Training Length: ', len(train_loader.dataset))
    
    print('Beginning Training')
    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model

    for epoch in range(max_epochs):
        print('\nEpoch: {}'.format(epoch))

        train(epoch, model, train_loader, optimizer)
        validation_loss = validate(epoch, model, valid_loader)
        
        if epoch - best_validation_epoch >= patience:
            break

        if validation_loss < best_validation_loss: # Select best model not from validation loss but validation accuracy
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)

        print(
            'Best validation at epoch {}, with avg. Log likelihood: {:.4f}'.
            format(best_validation_epoch, -best_validation_loss))

    # After training evaluate best model in terms of likelihood and prediction accuracy on
    # test data
    nll = validate(best_validation_epoch, best_model, test_loader, prefix='Test')
    ll = -nll
    bay_acc, hmm_acc, y_pred = evaluate(test_tensor, tst_y, trn_y, best_validation_epoch, best_model, log_priors, cond_inputs)
    
    if plot:
        plot_confMat(le, tst_y, y_pred)
    
    return bay_acc, hmm_acc, ll

# experiments = [['CARROTS', 16, False, 0, 0, False, False],
#                ['CARROTS', 16, False, 0, 0, False, True],
#                ['CARROTS', 16, False, 0, 0, True, False],
#                ['CARROTS', 16, False, 0, 0, True, True],
#                ['CARROTS', 16, True, 26, 13, False, False],
#                ['CARROTS', 16, True, 26, 13, False, True],
#                ['CARROTS', 16, True, 26, 13, True, False],
#                ['CARROTS', 16, True, 26, 13, True, True],
#                ['CARROTS', 16, True, 8, 4, True, False],
#                ['CARROTS', 16, True, 64, 32, True, False],
#                ['UCIHAR', 6, True, 0, 0, False, False]]

# experiments = [['MOSENSE', 6, True, 128, 64, False, False],
#                ['MOSENSE', 6, True, 128, 64, False, True],
#                ['MOSENSE', 6, True, 128, 64, True, False],
#                ['MOSENSE', 6, True, 128, 64, True, True],
#                ['MOSENSE', 6, True, 64, 32, True, True],
#                ['MOSENSE', 6, True, 32, 16, True, True],
#                ['UCIHAR', 6, True, 0, 0, False, False],
#                ['UCIHAR', 6, True, 0, 0, False, True]]


# df = pd.DataFrame(experiments, columns=['Dataset', '# classes', 'Window',
#                                         'W_size', 'W_step', 'Augment', 'Noise'])

# results = []

# for i in range(len(experiments)):
#     maf_acc, maf_ll = run(*experiments[i])
#     mvn_acc, mvn_ll = run_mvn(*experiments[i])
#     results.append({'MVN LL': mvn_ll, 'MAF LL': maf_ll, 'MVN ACC': mvn_acc, 'MAF ACC': maf_acc})
    
# rf = pd.DataFrame(results)

# df = pd.concat([df, rf], axis=1)
# df.to_csv('Experiment_results.csv', index=False)
# df.to_csv('Experiment_results_format.csv', float_format="%.4f", index=False)

# with open('resultstable.tex', 'w') as tf:
#       tf.write(df.to_latex(columns=['Dataset', 'Window', 'W_size', 'W_step',
#                                     'Augment', 'Noise', 'MVN LL', 'MAF LL', 'MVN ACC', 'MAF ACC'],
#                           index=False, float_format="%.4f"))

bay_acc, hmm_acc, ll = run('CARROTS', 16, False, 0, 0, False, False)
# bay_acc, hmm_acc, ll = run_mvn('CARROTS', 16, False, 0, 0, False, False)

print(bay_acc)
print(hmm_acc)
print(ll)