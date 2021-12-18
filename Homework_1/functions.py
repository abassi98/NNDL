### This file contains useful functions for the homework

# Test the network on test dataset to check accuracy on mnist classification
def my_accuracy(net, device, dataloader):
    """
    Compute the classification accuracy of the model
    ___________
    Parameters:
    net = network
    device = training device(cuda/cpu)
    dataloader = dataloader
    ________
    Returns:
    mismatched = list of all mismatched examples
    accuracy = classification accuracy
    """
    # Set evaluation mode
    network.eval()

    total = 0
    correct = 0
    mismatched = []
    
    with torch.no_grad():
        for  x_batch, label_batch in dataloader:
                
            # Move to device
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)

            # Forward pass
            y_hat = net(x_batch)
            y_hat = y_hat.squeeze()

            # Apply softmax 
            sf = nn.Softmax(dim=0)
            out_soft = sf(y_hat)

            # Take the prediction
            predicted = out_soft.detach().cpu().argmax().item()

            # True value
            true = label_batch.detach().cpu().item()

            if predicted==true:
                correct += 1
            else:
                mismatched.append((x_batch.detach().cpu().numpy(), predicted, true))
                                  
            total += 1

    return mismatched, 100.0*correct/total
    


