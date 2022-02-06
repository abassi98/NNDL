import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm_notebook
from functions import my_accuracy

### Train epoch function
def train_epoch(net, device, dataloader, loss_function, optimizer):
    """
    Train an epoch of data
    -----------
    Parameters:
    net = network
    device = training device (cuda/cpu)
    dataloader = dataloader of data
    loss_function = loss function
    optimzer = optimizer used
    --------
    Returns:
    mean(train_epoch_loss) = average epoch loss
    """
    # Set the train mode
    net.train()
    # List to save batch losses
    train_epoch_loss = []
    # Iterate the dataloader
    for x_batch, label_batch in dataloader:

        # Move to device
        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)
    
        # Forward pass
        y_hat = net(x_batch)

        # Compute loss
        loss = loss_function(y_hat, label_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute batch losses
        train_batch_loss = loss.detach().cpu().numpy()
        train_epoch_loss.append(train_batch_loss)
        
    return np.mean(train_epoch_loss)
    

### Test epoch function
def val_epoch(net,  device, dataloader, loss_function):
    """
    Validate an epoch of data
    -----------
    Parameters:
    net = network
    device = training device (cuda/cpu)
    dataloader = dataloader of data
    loss_function = loss function
    optimzer = optimizer used
    --------
    Returns:
    mean(val_epoch_loss) = average validation loss
    """
    # Set evaluation mode
    net.eval()
    # List to save evaluation losses
    val_epoch_loss = []
    with torch.no_grad():
        for x_batch, label_batch in dataloader:
                
            # Move to device
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)

            # Forward pass
            y_hat = net(x_batch)

            # Compute loss
            loss = loss_function(y_hat, label_batch)

            # Compute batch_loss
            val_batch_loss = loss.detach().cpu().numpy()
            val_epoch_loss.append(val_batch_loss)

    return np.mean(val_epoch_loss)

### Training epochs
def train_epochs(net, device, train_dataloader, val_dataloader, loss_function, optimizer, max_num_epochs, early_stopping = True):
    """
    Train an epoch
    ___________
    Parameters:
    max_num_epochs: maximum number of epochs (sweeps tthrough the datasets) to train the model
    early_stopping: if true stop the training if the last validation loss is greater 
    than the average of last 100 epochs
    """
    
    # Progress bar
    pbar = tqdm_notebook(range(max_num_epochs))

    # Inizialize empty lists to save losses
    train_loss_log = []
    val_loss_log = []

    for epoch_num in pbar:

        # Train epoch
        mean_train_loss = train_epoch(net, device, train_dataloader, loss_function, optimizer)

        # Validate epoch
        mean_val_loss = val_epoch(net, device, val_dataloader, loss_function)

        # Append losses and accuracy
        train_loss_log.append(mean_train_loss)
        val_loss_log.append(mean_val_loss)

        # Set pbar description
        pbar.set_description("Train loss: %s" %round(mean_train_loss,2)+", "+"Val loss %s" %round(mean_val_loss,2))
        
        # Early stopping
        if early_stopping:
            if epoch_num>10 and np.mean(val_loss_log[-10:]) < val_loss_log[-1]:
                print("Training stopped at epoch "+str(epoch_num)+" to avoid overfitting.")
                break
    
    return train_loss_log, val_loss_log

### Trainin epochs with accuracy (classification tasks)
def train_epochs_acc(net, device, train_dataloader, val_dataloader, test_dataloader, loss_function, optimizer, max_num_epochs, early_stopping = True):
    
    # Pogress bar
    pbar = tqdm_notebook(range(max_num_epochs))

    # Inizialize empty lists to save losses
    train_loss_log = []
    val_loss_log = []
    accuracy = []

    for epoch_num in pbar:
        # Compute accuracy before training
        mismatched, confusion, acc = my_accuracy(net, device, test_dataloader)

        # Tran epoch
        mean_train_loss = train_epoch(net, device, train_dataloader, loss_function, optimizer)

        # Validate epoch
        mean_val_loss = val_epoch(net,  device, val_dataloader, loss_function)

        # Append losses and accuracy
        train_loss_log.append(mean_train_loss)
        val_loss_log.append(mean_val_loss)
        accuracy.append(acc)

        # Set pbar description
        pbar.set_description("Train loss: %s" %round(mean_train_loss,2)+", "+"Val loss %s" %round(mean_val_loss,2)
                             +", "+"Test accuracy %s" %round(acc,2)+"%")
        
        # Early stopping
        if early_stopping:
            if np.mean(val_loss_log[-10:]) < val_loss_log[-1]:
                print("Training stopped at epoch "+str(epoch_num)+" to avoid overfitting.")
                break
    
    return train_loss_log, val_loss_log, accuracy

### Create k-folds splits
def KF_split(k_fold, batch_size, dataset):
    """
    Split dataset in k-folds for cross validation procedure
    ___________
    Parameters:
    k_folds: number of folds to divide train dataset
    dataset: dataset to be divided into k-folds during each epoch
    ________
    Returns:
    dataloaders: list containing all k_fold dataloaders after splitting
    """
    # Take the length of dataset
    len_dataset = len(dataset)
    
    # Define the k-folds
    len_fold = len_dataset//k_fold
    folds = torch.utils.data.random_split(dataset, k_fold*[len_fold])
    
    # Defin empty list dataloaders
    kf_dataloaders = []
    
    for f in range(k_fold):
        dataloader = DataLoader(folds[f], batch_size=batch_size, shuffle=True, num_workers=0)
        kf_dataloaders.append(dataloader)
        
    return kf_dataloaders


### Train epochs with k-fold cross validation
def kf_train_epochs(net, device, k_fold, batch_size, dataset, max_number_epochs, early_stopping = True):
    """
    """
    # Progress bar
    pbar = tqdm_notebook(range(max_num_epochs))

    # Inizialize empty lists to save fold list losses
    train_loss_log = []
    val_loss_log = []
    
    
    # Define dataloaders
    kf_dataloaders = KF_split(k_fold, batch_size, dataset)
    
    for epoch_num in pbar:
        # Empty lìsts to save fold losses
        train_loss_folds = []
        val_loss_folds = []
        
        # Iterate over each fold
        for f in range(k_fold):
            # Compute validation loss on f fold
            val_loss_fold = val_epoch(net, device, kf_dataloaders[f], loss_function)
            
            # Compute train loss on the other folds
            train_loss_fold = []
            
            for j in range(k_fold):
                if j != f:
                    # Train fold
                    mean_train_loss = train_epoch(net, device, kf_dataloaders[j], loss_function, optimizer)
                    train_loss_fold.append(mean_train_loss)
            
            train_loss_fold = np.mean(train_loss_fold)
            
            # Append in list for each epoch
            train_loss_folds.append(train_loss_fold)
            val_loss_folds.append(train_loss_fold)
            
        # Append fold losses lists on logs
        train_loss_log.append(train_loss_folds)
        val_loss_log.append(val_loss_folds)
        
    return train_loss_log, val_loss_log
