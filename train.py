import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(root = "data", download = True, train = True, transform = ToTensor())
train_data_set = DataLoader(train_data, 16)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 48, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(48, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 8) * (28 - 8), 10)
        )
    
    def forward(self, x):
        return self.model(x)

model = NN().to('cpu')
opt = Adam(model.parameters(), lr = 0.01)
CEL = nn.CrossEntropyLoss()

def main():
    for epochs in range(10):
        for batch in train_data_set:
            x, y = batch
            x = x.to('cpu')
            y = y.to('cpu')

            yhat = model(x)
            loss = CEL(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(model.state_dict, "cnn")


if __name__ == "__main__":
    main()