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

    with open("labels.csv", "w") as labels_file:
        labels_file.write("index,label\n")
        count = 0
        for index, label, pixels in parser():
            if label in ["0", "1"]:
                count += 1
                if count > 0 and count % 100 == 0:
                    print("Number of images: {}".format(count))
                labels_file.write(",".join([str(index), label]) + "\n")
                bitmap = np.array(pixels).reshape((28, 28))
                cv2.imwrite("train_images/{}.png".format(index), bitmap)

    print("You can see all the images with label 0 or 1 in train_images/")
    print("All the lables of training images are recored in labels.txt in order.")

if __name__ == "__main__":
    main()
