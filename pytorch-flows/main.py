import utils
import flows as fnn
import datasets
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

import sys
sys.path.insert(1, 'C:/Users/bened/PythonWork/CCBM_HAR/carrots/eval')
from loaders import *


if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')

flow = "maf"  # 'maf' or 'made'
dataset_name = "CARROTS"  # can be 'mnist', 'power', 'hepmass'
batch_size = 64 # 128 seems good
test_bs = 64 # can be 128
num_blocks = 5  # number of mades in maf
num_hidden = 180  # 1024 or 512 for mnist, 100 for power, 512 for hepmass
lr = 1e-5  # 5 works well
# random_order = False
patience = 10  # For early stopping
seed = 42
# plot = False
max_epochs = 50  # 1000
cond = True
no_cuda = False
num_cond_inputs = 16 # 16 for carrots, 8 for LARA, None if cond = False

cuda = not no_cuda and torch.cuda.is_available()
print('CUDA:', cuda)
device = torch.device("cuda:0" if cuda else "cpu")
print('Dveice:', device)

torch.manual_seed(seed)

#dataset = getattr(datasets, dataset_name)()

# For Carrots
print('--------Loading and Processing Data--------')
# trn_x, trn_y = load_conditional_train()
# v_x, v_y = load_conditional_val()
# tst_x, tst_y, le = load_conditional_test()
trn_x, trn_y, v_x, v_y, tst_x, tst_y, le = get_carrots()
troh_y = one_hot_encode(trn_y, num_cond_inputs)
vaoh_y = one_hot_encode(v_y, num_cond_inputs)
teoh_y = one_hot_encode(tst_y, num_cond_inputs)

# For LARA
# trn_x, trn_y = cond_lara_trn()
# v_x, v_y = cond_lara_val()
# tst_x, tst_y, le = cond_lara_tst()

# troh_y = one_hot_encode(trn_y, num_cond_inputs)
# vaoh_y = one_hot_encode(v_y, num_cond_inputs)
# teoh_y = one_hot_encode(tst_y, num_cond_inputs)


print('--------Correcting Data Type--------')
train_x = trn_x.astype('float32')
train_y = troh_y.astype('int')
val_x = v_x.astype('float32')
val_y = vaoh_y.astype('int')
test_x = tst_x.astype('float32')
test_y = teoh_y.astype('int')


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

#Tensor creation for unconditional datasets
# train_tensor = torch.from_numpy(dataset.trn.x)
# train_dataset = torch.utils.data.TensorDataset(train_tensor)

# valid_tensor = torch.from_numpy(dataset.val.x)
# valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

# test_tensor = torch.from_numpy(dataset.tst.x)
# test_dataset = torch.utils.data.TensorDataset(test_tensor)
# num_cond_inputs = None

print('Creating Data Loaders')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
# shuffling of training data is very important, because in the training data
# classes are grouped together due to temporal structure

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=test_bs,
    shuffle=False,
    drop_last=False)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=test_bs,
    shuffle=False,
    drop_last=False)


print('Initializing Model')
num_inputs = trn_x.shape[1]
# num_inputs = dataset.n_dims
act = 'tanh' if dataset_name == 'GAS' else 'relu'

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

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

writer = SummaryWriter(comment=flow + "_" + dataset_name)
global_step = 0

print('Training Length: ', len(train_loader.dataset))


def train(epoch):
    global global_step, writer
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
        # Test
        # Mhh gradient norms are pretty large, let's try gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if batch_idx % 500 == 0:
            print(f'Gradient Norm in batch {batch_idx} is {total_norm}')
        # Test
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))

        writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1

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
    global global_step, writer

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
            if batch_idx == 212:
                print(batch_log_probs)
            sum_log_prob = -corrected.sum().item()
            val_loss += sum_log_prob
            ##### In this sum the value inf appears some time find out why
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

    pbar.close()
    return val_loss / len(loader.dataset)

def test_predict(epoch, model, loader, prefix='Predict'):
    global global_step, writer

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

print('Beginning Training')
best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(max_epochs):
    print('\nEpoch: {}'.format(epoch))

    train(epoch)
    validation_loss = validate(epoch, model, valid_loader)

    if epoch - best_validation_epoch >= patience:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print(
        'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
        format(best_validation_epoch, -best_validation_loss))


validate(best_validation_epoch, best_model, test_loader, prefix='Test')

# run prediction by calculating p(x|y) for every class y and then Bayes
log_priors = [-4.02436356958844, -2.55757737723886, -2.62985841995197,
              -4.2595164232987, -2.49782943305708, -3.22986596419542,
              -2.48440939847247, -4.06681863851865, -2.16177315429147,
              -4.5156852300444, -5.62729572375705, -2.21563000487893,
              -5.27438211097532, -5.30111517317238, -2.03730365915227,
              -1.52852411531259]

lik = {}
N = test_x.shape[0]
print(f'N is {N}')
print(f'num_inputs is {num_inputs}')

for i in range(num_cond_inputs):
    # iterate through the classes to calculate p(x|y)
    # first one hot encode class to use for prediction
    a = np.full(N, i, dtype=int)
    pred_y = one_hot_encode(a, num_cond_inputs)
    pred_labels = torch.from_numpy(pred_y)
    pred_dataset = torch.utils.data.TensorDataset(test_tensor, pred_labels)

    pred_loader = torch.utils.data.DataLoader(
        pred_dataset,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False)

    log_pxy = test_predict(best_validation_epoch, best_model, pred_loader,
                            prefix='Test')

    lik[i] = log_pxy

result = np.zeros((N, num_cond_inputs))
for i in range(N):
    for c in range(num_cond_inputs):
        log_lik = lik[c][i]
        result[i, c] = log_lik + log_priors[c]

y_pred = np.argmax(result, axis=1)
np.savetxt("MAF_results.txt", result, fmt='%.5f', delimiter=" ")

print('accuracy: ', accuracy_score(tst_y, y_pred))

classes = list(le.classes_)
#classes = le
cf_matrix = confusion_matrix(tst_y, y_pred)
print('Number of test samples: ', np.sum(cf_matrix))
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                      columns = [i for i in classes])
plt.figure(figsize = (16,9))
s = sns.heatmap(df_cm, annot=True, cmap="flare", fmt='g')
s.set(xlabel='predicted class', ylabel='true class')
plt.savefig('conf_mat.jpg')