import os
import cv2
import numpy as np
from tqdm import tqdm  # Gives a progress bar
import torch
import torch.nn as nn
import torch.nn.functional as func
from abc import ABC

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
                    pass
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

        # print(x[0].shape)
        x = torch.flatten(x, 1, -1)  # Flatten
        if self._to_linear is None:
            # self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            self._to_linear = x.shape[1]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        return func.softmax(x, dim=1)


if __name__ == "__main__":
    if REBUILD_DATA:
        dogs_v_cats = DogsVsCats()
        dogs_v_cats.make_training_data()
    else:
        training_data = np.load("training_data.npy", allow_pickle=True)
        net = Net()
