import matplotlib.pyplot as plt
import matplotlib
import _tkinter as tk
from tensorflow import keras
import random
import numpy as np

img_size = 28
label_count = 10

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

test_data = np.loadtxt('data/mnist_test.csv', delimiter=',')

test_imgs = np.array(test_data[:, 1:])
test_labels = np.array(test_data[:, :1])

model = keras.models.load_model('model')

predictions = model.predict(test_imgs)

print('=======================================================================\n',
	  '                      	 											  \n',
	  '                              PROGRAM START                            \n',
	  '                                                                       \n',
	  '======================================================================')

count = input("How many numbers?: ")

matplotlib.use('TkAgg')

for i in range(int(count)):

    current_num = random.randint(0, len(test_imgs))

    plt.grid(False)
    plt.imshow(test_imgs[current_num].reshape((img_size, img_size)), cmap="Greys")
    prediction = np.argmax(predictions[current_num])
    plt.xlabel('Actual: ' + str(test_labels[current_num]))
    plt.title('Prediction: ' + str(labels[prediction]) + ' | Certanity: ' + str(predictions[current_num][prediction]))
    plt.show()
