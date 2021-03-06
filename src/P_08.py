import os
import cv2
import numpy as np
from tqdm import tqdm  # Gives a progress bar
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import time

from abc import ABC  # Imported to remove warning for class inheritance

REBUILD_DATA = False


class DogsVsCats:
    IMG_SIZE = 50
    CATS = "../PetImages/Cat"
    DOGS = "../PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []

    cat_count = 0
    dog_count = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                        if label == self.CATS:
                            self.cat_count += 1
                        elif label == self.DOGS:
                            self.dog_count += 1

                    except Exception as e:
                        print(f"Label: {label}")
                        print(f"FIle: {f}")
                        print(f"Exception: {str(e)}")

            np.random.shuffle(self.training_data)
            np.save("training_data.npy", self.training_data)
            print(f"Cats: {self.cat_count}")
            print(f"Dogs: {self.dog_count}")


class Net(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        # .Conv2d is input, output, kernel size
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        x = func.max_pool2d(func.relu(self.conv2(x)), (2, 2))
        x = func.max_pool2d(func.relu(self.conv3(x)), (2, 2))

        x = torch.flatten(x, 1, -1)  # Flatten

        if self._to_linear is None:
            # self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            self._to_linear = x.shape[1]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return func.softmax(x, dim=1)


def fwd_pass(x, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(x)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True) / len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss


# def test(size=32):
#     random_start = np.random.randint(len(test_X) - size)
#     x, y = test_X[random_start:random_start+size], test_Y[random_start:random_start+size]
#     with torch.no_grad():
#         val_acc, val_loss = fwd_pass(x.view(-1, 1, 50, 50).to(device), y.to(device))
#     return val_acc, val_loss
def test(size=32):
    X, y = test_X[:size], test_Y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1, 1, 50, 50).to(device), y.to(device))
    return val_acc, val_loss


def train(net):
    BATCH_SIZE = 100
    EPOCHS = 30  # Normally 1 - 10

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50).to(device)
                batch_y = train_Y[i:i+BATCH_SIZE].to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)
                if i % 50 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    if REBUILD_DATA:
        print("Building Data Set")
        dogs_v_cats = DogsVsCats()
        dogs_v_cats.make_training_data()

    print("Using Data Set")
    training_data = np.load("training_data.npy", allow_pickle=True)
    net = Net().to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
    X = X / 255
    Y = torch.Tensor([i[1] for i in training_data])

    VAL_PCT = 0.1
    val_size = int(len(X) * VAL_PCT)

    train_X = X[:-val_size]
    train_Y = Y[:-val_size]

    test_X = X[-val_size:]
    test_Y = Y[-val_size:]

    # This is where the network is trained and tested
    # val_acc, val_loss = test(size=32)
    # print(f"Accuracy: {val_acc}\nLoss: {val_loss}")

    MODEL_NAME = f"model-{int(time.time())}"
    print(MODEL_NAME)

    train(net)
