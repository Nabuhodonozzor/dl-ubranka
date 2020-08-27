import matplotlib.pyplot as plt
import matplotlib 
import _tkinter as tk
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random

matplotlib.use('TkAgg')

img_size = 28       # size of input image
label_count = 10    # number of numbers

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# loading data

train_data = np.loadtxt("data/mnist_train.csv", delimiter=',')
test_data = np.loadtxt('data/mnist_test.csv', delimiter=',')

# separating data into test and train

train_imgs = np.array(train_data[:, 1:])
train_labels = np.array(train_data[:, :1])

test_imgs = np.array(test_data[:, 1:])
test_labels = np.array(test_data[:, :1])

# adjusting data to fit the input neurons

train_imgs = train_imgs/255
test_imgs = test_imgs/255

# creating several neural networks 
# training and choosing the best one

best_acc = 0
for _ in range(10):

    print(f'Training {_ + 1}/10')

    # creating network

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(img_size, img_size)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(label_count, activation="softmax")
    ])

    # choosing parameters

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # training and determining best network

    model.fit(train_imgs, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_imgs, test_labels)

    print(best_acc)

    # saving network if it scores better

    if test_acc > best_acc:

        best_acc = test_acc
        print('New best network with score ', best_acc)
        model.save('model')

print('Best accuracy: ', best_acc)