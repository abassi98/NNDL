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
            self.act(),
	    nn.MaxPool2d(2),  # out = (N, 16, 12, 12)
	    nn.Dropout(self.drop_p),
            # Second convolution layer
	    nn.Conv2d(16, 32, 5), # out = (N, 32, 8, 8)
	    self.act(),
	    nn.MaxPool2d(2), # out = (N, 32, 4, 4)
	    nn.Dropout(self.drop_p)
	    )
    	
    	# Linear classifier
        self.lin = nn.Sequential(
            nn.Linear(in_features = 32*4*4, out_features = 128),
            self.act(),
            nn.Dropout(self.drop_p),
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

    def __init__(self, N_in, h1, h2, N_out):
        """
        Initialize a typical feedforward network with two hidden layers
        The input is typically a mnist image, given as a torch tensor of size = (1,784)
         ----------
        Parameters:
        in_size = size of visible layer
        out_size = size of output layers
        layer_sizes = list of sizes of hidden layers
        """

        super().__init__()

        # Parameters
        self.N_in = N_in
        self.h1 = h1
        self.h2 = h2
        self.N_out = N_out

        # Network architecture
        self.input = nn.Linear(in_features =N_in, out_features = h1)
        self.hidden = nn.Linear(in_features =h1, out_features = h2)      
        self.output = nn.Linear(in_features =h2, out_features = N_out)

        # Activation function      
        self.act = nn.ReLU()  
          
        print("Network initialized")
                  

    def forward(self, x):

        x = self.input(x)
        x = self.act(x)
        x = self.hidden(x)
        x = self.act(x)
        x = self.output(x)

        return x


























