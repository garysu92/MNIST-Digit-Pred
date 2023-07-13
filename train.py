from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(root = "data", download = True, train = True, transform = ToTensor())
train_data_set = DataLoader(train_data, 16)