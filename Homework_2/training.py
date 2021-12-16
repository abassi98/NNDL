import torch
import torch.nn as nn
import numpy as np

### Train function
def train_epoch(encoder, decoder, device, dataloader, loss_function, optimizer, noise = None):
    """
    Train an epoch of data
    -----------
    Parameters:
    encoder, decoder = network
    device = training device (cuda/cpu)
    dataloader = dataloader of data
    loss_function = loss function
    optimzer = optimizer used
    --------
    Returns:
    train_epoch_loss = list with all batch losses of the epoch
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

        # Set noise version equal to original image
        x_batch_noised = x_batch
    
        # Add noise
        if noise:
            x_batch_noised = noise(x_batch)
    
        # Encode data
        encoded_data = encoder(x_batch_noised)
        
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
    return train_epoch_loss
    

### Test function
def val_epoch(encoder, decoder,  device, dataloader, loss_function, noise = None):
    # Set evaluation mode
    encoder.eval()
    decoder.eval()
    # List to save evaluation losses
    val_epoch_loss = []
    with torch.no_grad():
        for x_batch, _ in dataloader:
                
            # Move to device
            x_batch = x_batch.to(device)

            # Set noise version equal to original image
            x_batch_noised = x_batch

            # Add noise
            if noise:
                x_batch_noised = noise(x_batch)

            # Encode data
            encoded_data = encoder(x_batch_noised)
 
            # Decode data
            decoded_data = decoder(encoded_data)

            # Compute loss
            loss = loss_function(decoded_data, x_batch)

            # Compute batch_loss
            val_batch_loss = loss.detach().cpu().numpy()
            val_epoch_loss.append(val_batch_loss)
    return val_epoch_loss
    

