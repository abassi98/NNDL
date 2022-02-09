import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm_notebook
import matplotlib.pyplot as plt
from IPython import display
from functions import my_accuracy

### Normal autoencoder training functions
# Train function
def train_epoch(net, device, dataloader, loss_function, optimizer, noise = None):
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
    net.train()
    
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
        encoded_data, decoded_data = net(x_batch)
        
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
def val_epoch(net, device, dataloader, loss_function, noise = None):
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
    net.eval()
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
            encoded_data, decoded_data = net(x_batch)

            # Compute loss
            loss = loss_function(decoded_data, x_batch)

            # Compute batch_loss
            val_batch_loss = loss.detach().cpu().numpy()
            val_epoch_loss.append(val_batch_loss)
            
    return np.mean(val_epoch_loss)

def train_epochs(net, device, train_dataloader, val_dataloader, test_data, loss_function, optimizer, max_num_epochs, early_stopping):
    """
    Train multiple epochs with early stopping
    --------
    Returns:
    list of train and validtion losses
    """
    
    train_loss_log = []
    val_loss_log = []
    
    # Define progeresso bar
    pbar = tqdm_notebook(range(max_num_epochs))

    for epoch_num in pbar:

        # Train an epoch and save losses
        mean_train_loss = train_epoch(net, device, train_dataloader, loss_function, optimizer)

        # Validate an epoch
        mean_val_loss = val_epoch(net, device, val_dataloader, loss_function)


        # Append to plot
        train_loss_log.append(mean_train_loss)
        val_loss_log.append(mean_val_loss)
        
        # Set pbar description
        pbar.set_description("Train loss: %s" %round(mean_train_loss,2)+","+"Validation loss %s" %round(mean_val_loss,2))
        
        # Early stopping
        if early_stopping:
            if epoch_num>10 and np.mean(val_loss_log[-10:]) < val_loss_log[-1]:
                print("Training stopped at epoch "+str(epoch_num)+" to avoid overfitting.")
                break

        ### Plot progress
        
        # Get the output of a specific image (the test image at index 0 in this case)
        img = test_data[0][0].unsqueeze(0).to(device)
        net.eval()
        with torch.no_grad():
            _ , rec_img  = net(img)
        # Plot the reconstructed image
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[0].set_title('Original image')
        axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[1].set_title('Reconstructed image (EPOCH %d)' % (epoch_num + 1))
        plt.tight_layout()
        plt.pause(0.1)
        # Save figures
        #os.makedirs('autoencoder_progress_%d_features' % encoded_space_dim, exist_ok=True)
        #fig.savefig('autoencoder_progress_%d_features/epoch_%d.jpg' % (encoded_space_dim, epoch_num + 1))
        fig.canvas.draw()
        #display.display(plt.show())
        #display.clear_output(wait=True)
    
    return train_loss_log, val_loss_log

    

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
        encoded_data, _ , _ = encoder(x_batch)

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
            encoded_data, _ , _ = encoder(x_batch)

            # Compute loss
            loss = loss_function(encoded_data, label_batch)

            # Compute batch_loss
            val_batch_loss = loss.detach().cpu().numpy()
            val_epoch_loss.append(val_batch_loss)
            
    return np.mean(val_epoch_loss)

def ft_train_epochs(net, device, train_dataloader, val_dataloader, test_dataloader, loss_function, optimizer, max_num_epochs, early_stopping):
    
    # Pogress bar
    pbar = tqdm_notebook(range(max_num_epochs))

    # Inizialize empty lists to save losses
    train_loss_log = []
    val_loss_log = []
    accuracy = []

    for epoch_num in pbar:
        
        # Compute accuracy before training
        mismatched, confusion, acc = my_accuracy(net.encoder, device, test_dataloader)

        # Tran epoch
        mean_train_loss = ft_train_epoch(net.encoder, device, train_dataloader, loss_function, optimizer)

        # Validate epoch
        mean_val_loss = ft_val_epoch(net.encoder,  device, val_dataloader, loss_function)

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
