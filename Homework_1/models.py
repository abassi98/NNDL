import torch
import torch.nn as nn


class ConvNet(nn.Module):
    
    def __init__(self, parameters):
        """
        The input is typically a MNIST images batch, encoded in a torch.tensor of size (N,1,28,28), where N is the batch size
        -----------
        Parameters:
        act = activation function 
        drop_p = dropout probability
        """
        super().__init__()
        
        # Parameters 
        self.act = parameters["act"]
        self.drop_p = parameters["drop_p"]
        
        
        ## Network architecture
        # Convolution part
        self.cnn = nn.Sequential(
            #first convolution layer
            nn.Conv2d(1, 16, 5),  # out = (N, 16, 24, 24)
            self.act(inplace = True),
        nn.MaxPool2d(2),  # out = (N, 16, 12, 12)
        nn.Dropout(self.drop_p, inplace = True),
            # Second convolution layer
        nn.Conv2d(16, 32, 5), # out = (N, 32, 8, 8)
        self.act(inplace = True),
        nn.MaxPool2d(2), # out = (N, 32, 4, 4)
        nn.Dropout(self.drop_p, inplace = True)
        )
    
        # Linear classifier
        self.lin = nn.Sequential(
            nn.Linear(in_features = 32*4*4, out_features = 128),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = True),
        nn.Linear(in_features = 128, out_features = 10)
        )

        print("Network initialized")
        
    def forward(self, x):
        # Convolution layer
        x = self.cnn(x)
        # Flatten layer
        x = torch.flatten(x, start_dim = 1)
        # Linear layer
        x = self.lin(x)
        return x
        
    
        
    
class FFNet(nn.Module):

    def __init__(self, parameters):
        """
        Initialize a typical feedforward network with different hidden layers
        The input is typically a mnist image, given as a torch tensor of size = (1,784)
         ----------
        Parameters:
        layers_sizes = list of sizes of the hidden layers, the first is the visible layer and the last is the output layer

        """

        super().__init__()

        # Parameters
        self.layers_sizes = parameters["layers_sizes"]
        self.num_layers = len(self.layers_sizes)
        self.act = parameters["act"]
        self.drop_p = parameters["drop_p"]
        
        # Network architecture
        layers = []
        for l in range(self.num_layers-2):
            layers.append(nn.Linear(in_features = self.layers_sizes[l], out_features = self.layers_sizes[l+1]))
            layers.append(nn.Dropout(self.drop_p, inplace = True))
            layers.append(self.act(inplace = True))
        
        layers.append(nn.Linear(in_features = self.layers_sizes[self.num_layers-2], out_features = self.layers_sizes[self.num_layers-1]))
        
        self.layers = nn.ModuleList(layers)
                          
        print("Network initialized")
                  

    def forward(self, x):

        for l in range(len(self.layers)):
            layer = self.layers[l]
            x = layer(x)
 
        return x


























