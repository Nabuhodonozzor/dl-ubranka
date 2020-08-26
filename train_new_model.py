from tensorflow import keras
import pickle

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255
test_images = test_images/255

best_acc = 0
for _ in range(15):

    print(f'Training {_ + 1}/15')

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=12)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print(best_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        model.save('model')

with open("test_images.pickle", "wb") as f_image:
    pickle.dump(test_images, f_image)

with open('test_labels.pickle', 'wb') as f_labels:
    pickle.dump(test_labels, f_labels)
