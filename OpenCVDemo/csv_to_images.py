#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import os

def parser():

    with open("train.csv") as rf:
        one_line = rf.readline().strip()

        for index, oneline in enumerate(rf):
            data = oneline.strip().split(",")
            label = data[0]
            pixels = map(int, data[1:])

            yield (index, label, pixels)

def main():
    if not os.path.exists("train_images"):
        os.makedirs("train_images")

    with open("train_images/labels.txt", "w") as labels_file:
        for index, label, pixels in parser():
            print("Processing {}-th image...".format(index))
            labels_file.write(label + "\n")
            bitmap = np.array(pixels).reshape((28, 28))
            cv2.imwrite("train_images/train_image_{}.png".format(index), bitmap)

    print("You can see all the images in train_images/")
    print("All the lables of training images are recored in train_images/labels.txt in order.")

if __name__ == "__main__":
    main()
