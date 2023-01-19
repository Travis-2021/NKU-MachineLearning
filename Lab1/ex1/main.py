# coding=utf-8
import numpy as np
import struct
import os

from data_process import load_mnist, load_data
from train import train
from evaluate import predict, cal_accuracy
    

if __name__ == '__main__':
    # initialize the parameters needed
    mnist_dir = "mnist_data/"
    train_data_dir = "train-images.idx3-ubyte"
    train_label_dir = "train-labels.idx1-ubyte"
    test_data_dir = "t10k-images.idx3-ubyte"
    test_label_dir = "t10k-labels.idx1-ubyte"
    k = 10
    iters = 500
    alpha = 1

    # get the data
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
    print("Got data. ")

    # train the classifier
    theta = train(train_images, train_labels, k, iters, alpha)
    print("Finished training. ") 

    # evaluate on the testset
    y_predict = predict(test_images, theta)
    accuracy = cal_accuracy(y_predict, test_labels)
    print('accuracy:', accuracy)
    print("Finished test. ") 

    # Result:
    # alpha     iter    acc     loss
    # 0.1       500     0.86    0.49
    # 0.25      100     0.82    0.62
    # 0.25      300     0.88    0.45

    # 0.3       100     0.84    0.57
    # 0.5       100     0.86    0.49
    # 0.7       100     0.87    0.46
    # 1         100     0.88    0.44    仅某几次不稳定
    # 1.2       100     0.868   0.49    前期不稳定
    # 1.5       100     0.83    0.70    不稳定
    # 2         100                     非常不稳定
    # 1         200     0.89    0.38
    # 1         500     0.91    0.31
    # 0.5       500     0.90    0.35
    # 1         1000    0.9148  0.286
    # 1.05      500     0.90    0.32    后期仍会颠簸
    #
    # OPT:
    # alpha=1   iter=500    acc=0.91    loss=0.31

