import torch
import torch.nn as nn
import numpy as np



### Train function
def train_epoch(network, device, dataloader, loss_function, optimizer):
    """
    Train an epoch of data
    -----------
    Parameters:
    network = model to be trained
    device = training device (cuda/cpu)
    dataloader = dataloader of data
    loss_function = loss function
    optimzer = optimizer used
    --------
    Returns:
    train_epoch_loss = list with all batch losses of the epoch
    """
    # Set the train mode
    network.train()
    # List to save batch losses
    train_epoch_loss = []
    # Iterate the dataloader
    for x_batch, label_batch in dataloader:
        # Move to device
        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)

        # Forward pass
        out = network(x_batch)

        # Compute loss
        loss = loss_function(out, label_batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute batch losses
        train_batch_loss = loss.detach().cpu().numpy()
        train_epoch_loss.append(train_batch_loss)
    return train_epoch_loss
    

### Test function
def val_epoch(network,  device, dataloader, loss_function):
    # Set evaluation mode
    network.eval()
    # List to save evaluation losses
    val_epoch_loss = []
    with torch.no_grad():
        for x_batch, label_batch in dataloader:     
            # Move to device
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)

            # Forward pass
            out = network(x_batch)

            # Compute loss
            loss = loss_function(out,label_batch)

            # Copmute batch_loss
            val_batch_loss = loss.detach().cpu().numpy()
            val_epoch_loss.append(val_batch_loss)
    return val_epoch_loss
    
    
"""
# Define the k-folds for cross-validation
num_fold = 7

kf = KFold(n_splits=num_fold, shuffle=True, random_state= 0)
for k, (train_index, test_index) in enumerate(kf.split(data_set)):
        
    
    train_data_loader = DataLoader(Subset(data_set, train_index.tolist()),
                                       batch_size= dm.train_dataloader().batch_size)
    test_data_loader = DataLoader(Subset(data_set, test_index.tolist()),
                                      batch_size=dm.train_dataloader().batch_size)

    
    trainer.fit(net, train_dataloader=train_data_loader)
    trainer.test(net, test_dataloaders=test_data_loader)

"""

#### OLd part of trainig in notebook  ####

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training device: {device}")

# Initialize Network and send to device
torch.manual_seed(10)
p = 0.1
net = ConvNet(p)
net.to(device)

# Define the loss function
loss_function = nn.CrossEntropyLoss()

# Define the optimizers
optimizer_1 = optim.Adam(net.parameters(), lr = 1e-4, weight_decay = 0.0)
optimizer_2 = optim.SGD(net.parameters(), lr = 1e-4, momentum = 1.4, weight_decay = 0.1)


### Training loop with progress bar

num_epochs = 50


train_loss_log = []
val_loss_log = []
accuracies = []

pbar = tqdm_notebook(range(num_epochs))

for epoch_num in pbar:
    
    # Train an epoch and save losses
    train_epoch_loss = train_epoch(net, device, train_dataloader, loss_function, optimizer_1)
    # Validate an epoch
    val_epoch_loss = val_epoch(net,  device, val_dataloader, loss_function)
    # Compute averages over an epoch
    mean_train_loss = np.mean(train_epoch_loss)
    mean_val_loss = np.mean(val_epoch_loss)
    
    # Test accuracy
    mismatched, acc = my_accuracy(net, device, test_dataloader)
    accuracies.append(acc)
    
    # Append to plot
    train_loss_log.append(mean_train_loss)
    val_loss_log.append(mean_val_loss)
    
    
    pbar.set_description("Train loss: %s" %round(mean_train_loss,3)+","+"Validation loss %s" %round(mean_val_loss,3))
    
    #sleep(0.03)
    pbar.update()
############################

# Plot losses
plt.figure(figsize=(12,8))
plt.semilogy(train_loss_log, label='Train loss')
plt.semilogy(val_loss_log, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()

#####################

# Plot test accuracy
plt.figure(figsize = (12,8))
plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy (%)')
plt.grid()
plt.legend()
plt.show()
print("Final accuracy: %d" %accuracies[-1])
print(accuracies)
