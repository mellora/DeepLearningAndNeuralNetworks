import os
import cv2
import numpy as np
from tqdm import tqdm  # Gives a progress bar

REBUILD_DATA = True


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
                    # print(str(e))
                    pass
            np.random.shuffle(self.training_data)
            np.save("training_data.npy", self.training_data)
            print(f"Cats: {self.cat_count}")
            print(f"Dogs: {self.dog_count}")


if __name__ == "__main__":
    if REBUILD_DATA:
        dogs_v_cats = DogsVsCats()
        dogs_v_cats.make_training_data()
