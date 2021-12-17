from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

class MNISTDataModule(LightningDataModule):
    def __init__(self, data_parameters):
        super().__init__()
        self.data_dir = data_parameters["data_dir"]
        self.batch_size = data_parameters["batch_size"]
        self.transform = transforms.Compose([transforms.ToTensor(),])

        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(mnist_full, [50000, 10000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

