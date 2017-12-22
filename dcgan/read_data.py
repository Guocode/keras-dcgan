# -*- coding: utf-8 -*-
import os 
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def read_data():
    X_train = input_data.read_data_sets("mnist",\
            one_hot=True).train.images
    X_train = X_train.reshape(-1, 28,\
            28, 1).astype(np.float32)
    X_test = input_data.read_data_sets("mnist",\
            one_hot=True).test.images
    X_test = X_test.reshape(-1, 28,\
            28, 1).astype(np.float32)        
    Y_train = input_data.read_data_sets("mnist",\
            one_hot=True).train.labels
    Y_test = input_data.read_data_sets("mnist",\
            one_hot=True).test.labels
    
    print((X_train.shape),(Y_train.shape),(X_test.shape),(Y_test.shape))

    
    return (X_train-0.5)/0.5,Y_train,X_test,Y_test

if __name__ == '__main__':
    X_train,_,_,_ = read_data()
    images = X_train[0:15,:,:,:]
    for i in range(16):
        #print(Y_train[i-1])
        plt.subplot(4, 4, i+1)
        image = images[i-1, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()