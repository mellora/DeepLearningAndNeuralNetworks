from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
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
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)    # Output 10 comes from the number of output classes 0 - 9

    def forward(self, x):   # Method defines how data travels between layers
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.fc4(x)
        # dim=0 would be for distribution of batch, dim=1 is for the output itself.  Default to dim=1
        return func.log_softmax(x, dim=1)


if __name__ == "__main__":
    net = Net()

    optimizer = optim.Adam(net.parameters(), lr=0.001)  # lr=0.001 can also be lr=1e-3
    
    EPOCHS = 3

    for epoch in range(EPOCHS):
        for data in train_set:
            # data is a batch of feature sets and Labels
            X, y = data
            net.zero_grad()
            output = net(X.view(-1, 28 * 28))
            # if output is a scalar value, use .nll_loss
            # if output is a one hot vector, use .mse_loss
            loss = func.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(loss)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_set:
            X, y = data
            output = net(X.view(-1, 28 * 28))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    print(f"Accuracy: {round(correct / total, 3)}")
