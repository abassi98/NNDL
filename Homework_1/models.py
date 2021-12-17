import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms
from torchmetrics import Accuracy

class ConvNet(LightningModule):
    
    def __init__(self, parameters):
        """
        The input is typically a MNIST images batch, encoded in a torch.tensor of size (N,1,28,28), where N is the batch size
        -----------
        Parameters:
        act = activation function 
        optmizer = optmizer used for backprop
        loss_fn = loss function used
        lr = learning rate
        L2_reg = weight decay/L2 regularization term 
        drop_p = dropout probability
        train_loss = list to save training loss
        val_loss = list to save validation loss
        val_acc = list to save validation accuracy
        """
        super().__init__()
        
        # Parameters 
        self.act = getattr(nn, parameters["act"])
        self.optimizer = getattr(optim, parameters["optimizer"])
        self.loss_fn = getattr(nn, parameters["loss_fn"])()
        self.lr = parameters["lr"]
        self.L2_reg = parameters["L2_reg"]
        self.drop_p = parameters["drop_p"]
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
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
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        self.train_loss.append(loss.item()) # Save train loss
        self.log("train_loss", loss, on_step = True, on_epoch = True,  prog_bar = True, logger = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        preds = torch.argmax(out, dim=1)
        acc = Accuracy()(preds, y)
        self.log('val_loss', loss, on_step = True, on_epoch = True, prog_bar=True, logger = True)
        self.log('val_acc', acc, on_step = True, on_epoch = True, prog_bar=True, logger = True)
        # Save loss and acc
        self.val_loss.append(loss.item())
        self.val_acc.append(acc.item())
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), lr = self.lr, weight_decay = self.L2_reg)
        return opt
        
    
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


























