import torch
import numpy as np
from maf import MAF
from made import MADE
from datasets.data_loaders import get_data, get_data_loaders
from utils.train import train_one_epoch_maf, train_one_epoch_made
from utils.validation import val_maf, val_made
from utils.test import test_maf, test_made
from utils.plot import sample_digits_maf, plot_losses

import sys
sys.path.insert(1, 'C:/Users/bened/PythonWork/CCBM_HAR/carrots/eval')
from loaders import load_dataset, load_config


# --------- SET PARAMETERS ----------
model_name = "maf"  # 'maf' or 'made'
dataset_name = "carrots" # can be 'mnist', 'power', 'hepmass'
batch_size = 128
n_mades = 5
hidden_dims = [512] #1024 or 512 for mnist, 100 for power, 512 for hepmass
lr = 1e-4
random_order = False
patience = 30  # For early stopping
seed = 290713
plot = False
max_epochs = 1000
# -----------------------------------

#TODO: Change this to accept the carrots data set and batch it correctly
# Get dataset. How do we handle the classes? Should do conditional density
# estimation as it perfroms better.

X_1, y_1,_ = load_dataset(1)
X_2, y_2,_ = load_dataset(2)
X_3, y_3,_ = load_dataset(3)
X_4, y_4,_ = load_dataset(4)
X_5, y_5,_ = load_dataset(5)
X_6, y_6,_ = load_dataset(6)
X_7, y_7,le = load_dataset(7)

X = [X_1, X_2, X_3, X_5, X_6]
train_x = np.concatenate(X, axis=0)
print('Train shape', train_x.shape)
val_x = X_4
test_x = X_7

train = torch.from_numpy(train_x)
val = torch.from_numpy(val_x)
test = torch.from_numpy(test_x)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)


# Get data loaders.
# Get model.
n_in = train.size(dim=1)
print('Number of dimensions:', n_in)
if model_name.lower() == "maf":
    model = MAF(n_in, n_mades, hidden_dims)
elif model_name.lower() == "made":
    model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True)
# Get optimiser.
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)


# Format name of model save file.
save_name = f"{model_name}_{dataset_name}_{'_'.join(str(d) for d in hidden_dims)}.pt"
# Initialise list for plotting.
epochs_list = []
train_losses = []
val_losses = []
# Initialiise early stopping.
i = 0
max_loss = np.inf
# Training loop.
for epoch in range(1, max_epochs):
    if model_name == "maf":
        train_loss = train_one_epoch_maf(model, epoch, optimiser, train_loader)
        val_loss = val_maf(model, train, val_loader)
    elif model_name == "made":
        train_loss = train_one_epoch_made(model, epoch, optimiser, train_loader)
        val_loss = val_made(model, val_loader)
    if plot:
        sample_digits_maf(model, epoch, random_order=random_order, seed=5)

    epochs_list.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Early stopping. Save model on each epoch with improvement.
    if val_loss < max_loss:
        i = 0
        max_loss = val_loss
        torch.save(
            model, "model_saves/" + save_name
        )  # Will print a UserWarning 1st epoch.
    else:
        i += 1

    if i < patience:
        print("Patience counter: {}/{}".format(i, patience))
    else:
        print("Patience counter: {}/{}\n Terminate training!".format(i, patience))
        break

plot_losses(epochs_list, train_losses, val_losses, title=None)
