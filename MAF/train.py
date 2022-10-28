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
from loaders import *


# --------- SET PARAMETERS ----------
model_name = "maf"  # 'maf' or 'made'
dataset_name = "lara" # can be 'mnist', 'power', 'hepmass'
batch_size = 128
n_mades = 5
hidden_dims = [100] #1024 or 512 for mnist, 100 for power, 512 for hepmass
lr = 1e-3 #5 works well
random_order = False
patience = 5  # For early stopping
seed = 290713
plot = False
max_epochs = 15 #1000
# -----------------------------------

#train dict is a dictionary of pandas dataframes obtained from the original train data
#via grouping by class, for conditional density estimation. The keys are the classes
#(integers 0 to 15) and the value is the dataframe containing all examples of this class

#CARROTS
train_dict,_ = load_split_train()
val_dict,_ = load_split_val()

# LARA
# train_dict,_ = load_lara_trn()
# val_dict,_ = load_lara_val()



cc = len(train_dict.keys()) # classs count (i.e. number of distict classes)
for j in range(cc): # Train a MAF per each of 16 classes

    train_x = train_dict[j].to_numpy()
    val_x = val_dict[j].to_numpy()
    
    train = torch.from_numpy(train_x)
    val = torch.from_numpy(val_x)
    #test = torch.from_numpy(test_x)
    
    # Get data loaders.
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,)
    #test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,)
    
    
    # Get model.
    n_in = train.size(dim=1)
    print('Number of dimensions:', n_in)
    if model_name.lower() == "maf":
        model = MAF(n_in, n_mades, hidden_dims)
    elif model_name.lower() == "made":
        model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True)
    # Get optimiser.
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6) #decay was -6
    
    
    # Format name of model save file.
    save_name = f"{model_name}_{dataset_name}_{j}_{'_'.join(str(d) for d in hidden_dims)}.pt"
    # Initialise list for plotting.
    epochs_list = []
    train_losses = []
    val_losses = []
    # Initialiise early stopping.
    i = 0
    max_loss = np.inf
    # Training loop.
    print(f'Starting training for class {j}')
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
    
    print(f'Ending training for class {j}')
    #plot_losses(epochs_list, train_losses, val_losses, title=None)
