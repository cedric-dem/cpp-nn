# C++ Single Layer Perceptron

## Architecture
neural network without hidden layer. the weight matrix is a matrix of size 784x10


## Dataset
source : https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

split into 

	train : 60 000 images 

	test : 10 000 images

each of size 28x28 (grayscale)

representing digits between 0 and 9

## Before executing
download both train and test in csv format, should be located in the dataset folder 

then make

## Train

./train

will use the train dataset to train a model, then put the resulting weight matrix in a separate file

## Test

./test

will use the test dataset to evaluate the model produced by the train script
