import torch
import torch.nn as nn
from VAE_functions import Sampler

class ConvEncoder(nn.Module):
    
    def __init__(self, parameters):
        
        super().__init__()
        
        # Retrieve parameters
        self.encoded_space_dim = parameters["encoded_space_dim"]
        self.drop_p = parameters["drop_p"]
        self.act = parameters["act"]
        
        ### Network architecture
        # First convolutional layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, 5),  # out = (N, 16, 24, 24)
            nn.BatchNorm2d(16),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.MaxPool2d(2, return_indices=True)  # out = (N, 16, 12, 12)
        )
        
        # Second convolution layer
        self.second_conv = nn.Sequential(
            nn.Conv2d(16, 32, 5), # out = (N, 32, 8, 8)
            nn.BatchNorm2d(32),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.MaxPool2d(2, return_indices=True) # out = (N, 32, 4, 4)
        )
        
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        # Linear encoder
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(32*4*4, 128),
            nn.BatchNorm1d(128),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            # Second linear layer
            nn.Linear(128, self.encoded_space_dim)
        )

        
    def forward(self, x):
        # Apply first convolutional layer
        x, indeces_1 = self.first_conv(x)
        
        # Apply second convolutional layer
        x, indeces_2 = self.second_conv(x)
        
        # Flatten 
        x = self.flatten(x)
        
        # Apply linear encoder layer
        encoded_data = self.encoder_lin(x)
        
        return (encoded_data, indeces_1, indeces_2)
    
class ConvDecoder(nn.Module):
    
    def __init__(self, parameters):
        
        super().__init__()
        
        # Retrieve parameters
        self.encoded_space_dim = parameters["encoded_space_dim"]
        self.drop_p = parameters["drop_p"]
        self.act = parameters["act"]
        
        ### Network architecture
        # Linear decoder
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(self.encoded_space_dim, 128),
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = False),
            nn.BatchNorm1d(128),
            # Second linear layer
            nn.Linear(128, 32*4*4)
        )
        
        # Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 4, 4))
        
        # Unpooling layer
        self.unpool = nn.MaxUnpool2d(2)
        
        self.first_deconv = nn.Sequential(
            # First transposed convolution
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 5) # out = (N ,16, 12, 12)    
        )
        
        self.second_deconv = nn.Sequential(
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 5) # out = (N, 1, 28, 28)
        )
        
    def forward(self, x, indeces_1, indeces_2):
        
        # Apply linear decoder
        x = self.decoder_lin(x)
        
        # Unflatte layer
        x = self.unflatten(x)
        
        # Apply first unpooling layer
        x = self.unpool(x,  indeces_2)
        
        # Apply first deconvolutional layer
        x = self.first_deconv(x)
        
        # Apply second unpooling layer
        x = self.unpool(x, indeces_1)
        
        # Apply second deconvolutional layer
        x = self.second_deconv(x)
        
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        decoded_data = torch.sigmoid(x)
        
        return decoded_data
        

class ConvAE(nn.Module):
    
    def __init__(self, parameters):
        super().__init__()
        
        self.encoder = ConvEncoder(parameters)
        self.decoder = ConvDecoder(parameters)
        
        
    def forward(self, x):
        
        # Encode data and keep track of indexes
        encoded_data, indeces_1, indeces_2 = self.encoder(x)
        
        # Decode data
        decoded_data = self.decoder(encoded_data, indeces_1, indeces_2)
        
        return (encoded_data, decoded_data)




class VConvEncoder(nn.Module):

    def __init__(self, encoded_space_dim):
    
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        
        # Encode mean
        self.encoder_mean = nn.Sequential(
            # First linear layer
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(64, encoded_space_dim)
        )
        
        # Encode log_var
        
        self.encoder_logvar = nn.Sequential(
            # First linear layer
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(64, encoded_space_dim)
        )
        
    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        ## Apply linear layers
        # Encode mean
        mean = self.encoder_mean(x)
        # Encode log_var
        log_var = self.encoder_logvar(x)
        
        return (mean, log_var)
	
class ConvVAE(nn.Module):
    
    def __init__(self, encoded_space_dim):
    
        super().__init__()
        
        self.encoder = VConvEncoder(encoded_space_dim)
        self.decoder = ConvDecoder(encoded_space_dim)
        
        
    def forward(self, x):
    
        ### Encode data       
        mean, log_var = self.encoder(x)
        
        # Sampling
        x = Sampler(mean, log_var)
        
        ### Decode data       
        x = self.decoder(x)
        
        return (mean, log_var, x)
        
    
        
      
    
      
      
      
      
      
      
 
