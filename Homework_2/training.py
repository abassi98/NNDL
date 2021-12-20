import torch
import torch.nn as nn
import numpy as np

### Normal autoencoder training functions
# Train function
def train_epoch(encoder, decoder, device, dataloader, loss_function, optimizer, noise = None):
    """
    Train an epoch of data
    -----------
    Parameters:
    encoder, decoder: network
    device: training device (cuda/cpu)
    dataloader: dataloader of data
    loss_function: loss function
    optimzer: optimizer used
    noise: noise to be added in denoising version
    --------
    Returns:
    average training loss over the epoch
    """
    # Set the train mode
    encoder.train()
    decoder.train()
    # List to save batch losses
    train_epoch_loss = []
    # Iterate the dataloader
    for x_batch, _ in dataloader:
        # Move to device
        x_batch = x_batch.to(device)
    
        # Add noise
        if noise:
            x_batch = noise(x_batch)
    
        # Encode data
        encoded_data = encoder(x_batch)
        
        # Decode data
        decoded_data = decoder(encoded_data)

        # Compute loss
        loss = loss_function(decoded_data, x_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute batch losses
        train_batch_loss = loss.detach().cpu().numpy()
        train_epoch_loss.append(train_batch_loss)
        
    return np.mean(train_epoch_loss)
    

# Validation function
def val_epoch(encoder, decoder,  device, dataloader, loss_function, noise = None):
    """
    Validate an epoch of data
    -----------
    Parameters:
    encoder, decoder: network
    device: training device (cuda/cpu)
    dataloader: dataloader of data
    loss_function: loss function
    noise: noise to be added in denoising version
    --------
    Returns:
    average validation loss over the epoch
    """
    # Set evaluation mode
    encoder.eval()
    decoder.eval()
    # List to save evaluation losses
    val_epoch_loss = []
    with torch.no_grad():
        for x_batch, _ in dataloader:
                
            # Move to device
            x_batch = x_batch.to(device)

            # Add noise
            if noise:
                x_batch = noise(x_batch)

            # Encode data
            encoded_data = encoder(x_batch)
 
            # Decode data
            decoded_data = decoder(encoded_data)

            # Compute loss
            loss = loss_function(decoded_data, x_batch)

            # Compute batch_loss
            val_batch_loss = loss.detach().cpu().numpy()
            val_epoch_loss.append(val_batch_loss)
            
    return np.mean(val_epoch_loss)
    

    

### Fine tuning
# Train function
def ft_train_epoch(encoder, device, dataloader, loss_function, optimizer, noise = None):
    """
    Fine tune the encoder with a supervised task
    -----------
    Parameters:
    encoder: network
    device: training device (cuda/cpu)
    dataloader: dataloader of data
    loss_function: loss function
    optimzer: optimizer used
    noise: noise to be added in denoising version
    --------
    Returns:
    average training loss over the epoch
    """
    # Set the train mode
    encoder.train()
  
    # List to save batch losses
    train_epoch_loss = []
    
    # Iterate the dataloader
    for x_batch, label_batch in dataloader:
        # Move to device
        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)
    
        # Add noise
        if noise:
            x_batch = noise(x_batch)
    
        # Encode data
        encoded_data = encoder(x_batch)

        # Compute loss
        loss = loss_function(encoded_data, label_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute batch losses
        train_batch_loss = loss.detach().cpu().numpy()
        train_epoch_loss.append(train_batch_loss)
        
    return np.mean(train_epoch_loss)


# Validation function
def ft_val_epoch(encoder,  device, dataloader, loss_function, noise = None):
    """
    Validation function for fine tuning
    -----------
    Parameters:
    encoder: network
    device: training device (cuda/cpu)
    dataloader: dataloader of data
    loss_function: loss function
    noise: noise to be added in denoising version
    --------
    Returns:
    average validation loss over the epoch
    """
    # Set evaluation mode
    encoder.eval()

    # List to save evaluation losses
    val_epoch_loss = []
    with torch.no_grad():
        for x_batch, label_batch in dataloader:
                
            # Move to device
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)

            # Add noise
            if noise:
                x_batch = noise(x_batch)

            # Encode data
            encoded_data = encoder(x_batch)

            # Compute loss
            loss = loss_function(encoded_data, label_batch)

            # Compute batch_loss
            val_batch_loss = loss.detach().cpu().numpy()
            val_epoch_loss.append(val_batch_loss)
            
    return np.mean(val_epoch_loss)
