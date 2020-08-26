from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle

p_test_images = open('test_images.pickle', 'rb')
test_images = pickle.load(p_test_images)

p_test_labels = open('test_labels.pickle', 'rb')
test_labels = pickle.load(p_test_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.models.load_model('model')

prediction = model.predict(np.array(test_images))

for i in range(len(test_images)):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
