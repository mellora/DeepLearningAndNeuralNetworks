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
    # plt.imshow(data[0][0].view(28, 28))
    # plt.show()
    total = 0
    counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    for data in train_set:
        Xs, Ys = data
        for y in Ys:
            counter_dict[int(y)] += 1
            total += 1
    print(counter_dict)
    print(total)
    for i in counter_dict:
        print(f"{i}: {counter_dict[1] / total * 100}")
