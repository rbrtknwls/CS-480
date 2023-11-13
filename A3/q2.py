import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import layers, models


# ==================== Import MNIST dataset and Normalize ====================

def normalize_img(image):
    scaled_image = image / 255.  # Scale the image to 0-1

    # Add the top padding to the image
    top = np.zeros((2, 28))
    bottom = top
    image32By28 = np.concatenate([top, scaled_image, bottom], 0)

    left = np.zeros((32, 2))
    right = left
    finalImage = np.concatenate([left, image32By28, right], 1)

    return finalImage  # Return image with label


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


updated_train = []
updated_test = []
for i in range(0, len(x_train)):
    updated_train.append(normalize_img(x_train[i]))

for i in range(0, len(x_test)):
    updated_test.append(normalize_img(x_test[i]))

updated_train = np.array(updated_train)
updated_test = np.array(updated_test)

# ==================== Model Definition ====================

model = models.Sequential()

model.add(
    layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation='relu', input_shape=(32, 32, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(layers.Flatten())

model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(.5))

model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(.5))

model.add(layers.Dense(10))

# ==================== Training and Testing ====================

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

print(model.summary())

trainingAccuracy = []
trainingLoss = []
for i in range(0, 1):
    history = model.fit(updated_train, y_train, epochs=1, batch_size=512)

    trainingAccuracy.append((history.history["sparse_categorical_accuracy"], 1+i))
    trainingLoss.append((history.history["loss"], 1+i))

    model.save('q3Trained'+str(i)+'.keras')


plt.plot(trainingAccuracy)
plt.plot(trainingLoss)
plt.title('Accuracy vs # of Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid()
plt.show()

'''
new_model = tf.keras.models.load_model('q3Trained.keras')


predicted = new_model.predict(updated_test, verbose=1)
print(predicted)
predicted = np.argmax(predicted, axis=1)[:100]
label = y_test[:100]

print("Predicted label: ", *predicted)
print("True label     : ", *label)
'''