#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import random

"""
Simple Back-Propagation Neural Network.

Basic Usage:

X_train = np.array([[1, 1, 3],
                    [2, 1, 5],
                    [3, 1, 2]]) # 3x3 array

Y_train = np.array([[1],
                    [6],
                    [11]]) # 3x1 array

nn = NeuralNetwork([3, 10, 1]) # a 3x10x1 neural network

# Training
nn.train(X_train, Y_train)

X_test = np.array([[1, 2, 4]])

# Prediction
nn.eval(X_test)

"""

class NeuralNetwork(object):
    def __init__(self, nnStruct, r = 0.1):
        """
        nnStruct <list>: a list which specify the structure of the neural network
        r <float>: the range of random initiation of the weights of links.

        eg.
        nn = NeuralNetwork([2, 6, 1])
        # `nn` is a 2x6x1 neural network
        """
        self.struct = nnStruct
        self.layers = []
        self.weights = [None] # None is a place holder, links[i] are links from layer i-1 to i
        for layerNodes in nnStruct:
            tmp = np.zeros(layerNodes+1)
            tmp[0] = 1.   # the "constant term" node
            self.layers.append(tmp)

        for layerID in range(1, len(nnStruct) ):
            self.weights.append((np.random.rand(nnStruct[layerID], nnStruct[layerID-1]+1)*2. - 1.) * r)

    def __str__(self):

        return "NeuralNetwork({})".format(self.struct)

    def eval(self, input):
        """
        get the prediction form NN with the input

        input: np array of correct size
        """
        if len(input)!=self.struct[0] :
            print("Wrong input shape for NN: {}".format(input.shape))
            return None

        self.layers[0][1:] = input
        for layerID in range(1, len(self.struct)):
            self.layers[layerID][1:] = self.weights[layerID].dot(self.layers[layerID-1])
            self.layers[layerID][1:] = np.tanh(self.layers[layerID][1:])

        return self.layers[-1][1:]

    def train(self, train_X, train_Y, eta = 0.1, num_iteration = 1000):
        """
        train the NN
        train_X : the input, an NxM array.
        train_Y : the target values, an NxK array.
        eta     : the step size to be used in the gradient decend
        """
        N = train_X.shape[0]

        for iter_index in range(num_iteration):
            if iter_index > 0 and (iter_index + 1) % 100 == 0:
                print("Training progress: {}% done".format(100.0 * (iter_index + 1) / num_iteration))
            random_index = random.sample(range(N), 1)[0]
            input = train_X[random_index]
            answer = train_Y[random_index]
            output = self.eval(input)
            delta = []
            for i in self.struct:
                delta.append(np.zeros(i))

            delta[-1] = -2.*( answer - output ) * (1.-pow(output,2))
            for layerID in range(len(self.struct)-2,0,-1):
                delta[layerID] = self.weights[layerID+1][:,1:].transpose().dot(delta[layerID+1]) * (1.-pow(self.layers[layerID][1:],2))

            for layerID in range(1,len(self.struct)):
                self.weights[layerID] -= eta * np.outer(delta[layerID], self.layers[layerID-1])

