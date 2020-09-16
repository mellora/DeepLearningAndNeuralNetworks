import torch
import torchvision
from torchvision import transforms, datasets

# Data
train = datasets.MNIST("..", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("..", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

if __name__ == "__main__":
    pass
