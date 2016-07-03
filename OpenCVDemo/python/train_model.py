#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import os
import numpy as np
import random
from SimpleNN import NeuralNetwork
import argparse

def get_train_data(label_file, image_dir, num_samples = 1000):

    with open(label_file) as rf:
        rf.readline() # get rid of header.
        lines = map(lambda l: l.strip(), rf.readlines())

        sample_tuples = map(lambda s: s.split(","), lines)

        random_indices = random.sample(range(len(lines)), num_samples)
        random_samples = []

        for index in random_indices:
            random_samples.append(sample_tuples[index])

        image_index, label = random_samples[0]

        train_X = cv2.imread(os.path.join(image_dir, "{}.png".format(image_index)), 0).flatten()
        hot_encode = [0.0 for _ in range(2)]
        hot_encode[int(label)] = 1.0;
        train_Y = np.array(hot_encode)

        for sample in random_samples[1:]:
            image_index, label = sample
            bitmap = cv2.imread(os.path.join(image_dir, "{}.png".format(image_index)), 0)
            train_X = np.vstack([train_X, bitmap.flatten()])

            hot_encode = [0.0 for _ in range(2)]
            hot_encode[int(label)] = 1.0;
            target = np.array(hot_encode)
            train_Y = np.vstack([train_Y, target])

    return train_X, train_Y


def main(nnStructure):
    train_X, train_Y = get_train_data("labels.csv", "train_images", 3000)
    nn = NeuralNetwork(nnStructure)
    print("training {}".format(nn))
    nn.train(train_X, train_Y, 5000)

    with open("model.txt", "w") as wf:

        wf.write("NumberOfLayers: {}\n".format(len(nn.layers)))
        wf.write("NetworkStructure: {}\n".format(" ".join(map(str, nn.struct))))

        for weight in nn.weights[1:]:
            wf.write(" ".join(map(str, weight.flatten())) + "\n")


def network_structure_type(arg_str):

    structure = list(map(int, arg_str.strip().split(",")))

    return structure

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--structure", default = "784,1400,700,2",
                        help = "the network structure. ex: 784,1400,700,2 (a 784x1400x700x2 network, default)",
                        type = network_structure_type,
                        dest = "structure")

    args = parser.parse_args()
    main(args.structure)
