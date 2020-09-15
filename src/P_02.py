import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Data
train = datasets.MNIST("..", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("..", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

if __name__ == "__main__":
    for data in train_set:
        # print(data)
        break
    # x, y = data[0][0], data[1][0]
    # print(y)
    # print(x)
    # print(data[0][0].shape)
    plt.imshow(data[0][0].view(28, 28))
    plt.show()
