import os
import cv2
import numpy as np
from tqdm import tqdm  # Gives a progress bar
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from abc import ABC  # Used by class inheritance

REBUILD_DATA = False

BATCH_SIZE = 100    # Modify if memory issues happen
EPOCHS = 10


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


def train(network):
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_x = train_X[i: i + BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_Y[i: i + BATCH_SIZE]

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            network.zero_grad()

            outputs = network(batch_x)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}.  Loss: {loss}")


def test(network):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_Y[i]).to(device)
            net_out = network(test_X[i].view(-1, 1, 50, 50).to(device))[0]

            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print(f"Accuracy: {round(correct / total, 3)}")


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
    else:
        print("Using Data Set")
        training_data = np.load("training_data.npy", allow_pickle=True)
        net = Net().to(device)

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
        train(net)
        test(net)
