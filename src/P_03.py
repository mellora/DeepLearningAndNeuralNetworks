from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
from torchvision import transforms, datasets

# Data
train = datasets.MNIST("..", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("..", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class Net(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # Remember to include super().__init__() when inheriting
        # INPUT: 784 is equal to 28*28, gotten from the image size of the input
        # OUTPUT: 64 is the target number of neurons per layer
        # self.fc1 = nn.Linear(784, 64)
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)    # Output 10 comes from the number of output classes 0 - 9

    def forward(self, x):
        # Method defines how data travels between layers
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.fc4(x)
        # Following code works, but probably will not scale properly. Lacking activation function.
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        return x


if __name__ == "__main__":
    net = Net()
    print(net)
