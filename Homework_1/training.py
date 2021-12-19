import torch
import torch.nn as nn
import numpy as np


### Train function
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
    

### Test function
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


