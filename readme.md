# Neural Net

## Overview

This project is a basic implementation of a neural network based primarily on the code and ideas presented in the online book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen.

The primary goals of this project are:

    - To understand and internalize the basic concepts behind neural networks, the backpropagation alogrithm, and deep learning.
    - To create an object-oriented implementation of a neural network with an intuitive API in a language that is less commonly used for deep learning.

## Requirements to run this code

### Environment

This project targets .Net Core version 2.1. In order to build and run the project you'll need to download and install the [SDK from Microsoft](https://dotnet.microsoft.com/download/dotnet-core/2.1)

### Data

The project currently outputs an executable where the network structure, training data, and test data are all explicitly defined in the file `Network/Program.cs`.

The current configuration builds a network with a single hidden layer of 30 neurons. The network is trained and tested with the training and testing images from the MNIST dataset.

In order to use this file as-is, you will need to download the 4 data files (images and labels for both training and test sets) which you can get from [The MNIST Database](http://yann.lecun.com/exdb/mnist/). Unzip the files to `Network/data/`.

### Running the program

```shell
$ cd Network
$ dotnet build
$ dotnet run
```