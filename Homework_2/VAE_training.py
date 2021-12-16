import torch
import torch.nn as nn
from VAE_functions import Sampler, nKLDivLoss


### Train function
def train_epoch(net, device, dataloader, beta, loss_function, optimizer):
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
    train_epoch_loss = list with all batch losses of the epoch
    """
    # Set the train mode
    net.train()
    # List to save batch losses
    train_div_loss = []
    train_rec_loss = []
    # Iterate the dataloader
    for x_batch, _ in dataloader:
            
        # Move to device
        x_batch = x_batch.to(device)

        # Forward pass
        mean, log_var, decoded_data = net(x_batch)

        # Compute losses
        KLDiv_loss = nKLDivLoss(mean, log_var)
        MSE_loss = loss_function(decoded_data, x_batch)
        loss = MSE_loss + beta*KLDiv_loss 
      
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute batch losses
        KLDiv_loss = KLDiv_loss.detach().cpu().numpy()
        MSE_loss = MSE_loss.detach().cpu().numpy()
        train_div_loss.append(KLDiv_loss)
        train_rec_loss.append(MSE_loss)
        
    return (train_div_loss, train_rec_loss)
    

### Test function
def val_epoch(net,  device, dataloader, loss_function):
   # Set evaluation mode
   net.eval()
   # List to save evaluation losses
   val_div_loss = []
   val_rec_loss = []
   with torch.no_grad():
      for x_batch, _ in dataloader:
                
         # Move to device
         x_batch = x_batch.to(device)

         # Forward pass
         mean, log_var, decoded_data = net(x_batch)
        
         # Compute losses
         KLDiv_loss = nKLDivLoss(mean, log_var)
         MSE_loss = loss_function(decoded_data, x_batch)
         
         # Compute batch losses
         KLDiv_loss = KLDiv_loss.detach().cpu().numpy()
         MSE_loss = MSE_loss.detach().cpu().numpy()
         val_div_loss.append(KLDiv_loss)
         val_rec_loss.append(MSE_loss)
         
   return (val_div_loss, val_rec_loss)


