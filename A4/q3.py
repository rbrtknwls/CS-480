import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import layers, models


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


lossObject = tf.keras.losses.CategoricalCrossentropy()


def generate_adversarial_image(image, label, model):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = lossObject(label, prediction)

    gradient = tape.gradient(loss, image)
    return np.array(gradient[0])


trainedModel = tf.keras.models.load_model('../Midterm/MINSTtrained.keras')

probModel = tf.keras.Sequential([
  trainedModel,
  layers.Softmax()
])

probModel.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



updated_test_1 = []
updated_test_2 = []
updated_test_5 = []

for i in range(0, len(y_test)):
    norm = np.expand_dims(normalize_img(x_test[i]), axis=0)
    norm_as_tensor = tf.convert_to_tensor(norm)
    sparseLabel = []
    print(i)
    if i % len(y_test)/10 == 0:
        print("tenth")

    for x in range(0, 10):
        if x == y_test[i]:
            sparseLabel.append(1.0)
        else:
            sparseLabel.append(0)

    sparseLabel = tf.convert_to_tensor(np.expand_dims(sparseLabel, axis=0))

    delta = generate_adversarial_image(norm_as_tensor, sparseLabel, probModel)
    signedDelta = np.sign(delta)
    updated_test_1.append(np.add(norm[0], signedDelta*0.1))
    updated_test_2.append(np.add(norm[0], signedDelta*0.2))
    updated_test_5.append(np.add(norm[0], signedDelta*0.5))


updated_test_1 = np.array(updated_test_1)
updated_test_2 = np.array(updated_test_2)
updated_test_5 = np.array(updated_test_5)

ev1 = np.array(probModel.evaluate(updated_test_1, y_test))
ev2 = np.array(probModel.evaluate(updated_test_2, y_test))
ev5 = np.array(probModel.evaluate(updated_test_5, y_test))

np.savetxt("ev1.csv", ev1, delimiter=",")
np.savetxt("ev2.csv", ev2, delimiter=",")
np.savetxt("ev5.csv", ev5, delimiter=",")

'''
for imageIndex in range(0,3):
    for epsilonChoice in range(0, 3):
        plt.subplot(3, 3, imageIndex*3+epsilonChoice+1)
        if epsilonChoice == 0:
            plt.imshow(updated_test_1[imageIndex])
            plt.title("Epsilon 0.1 for " + str(y_test[imageIndex]))
        elif epsilonChoice == 1:
            plt.imshow(updated_test_2[imageIndex])
            plt.title("Epsilon 0.2 for " + str(y_test[imageIndex]))
        else:
            plt.imshow(updated_test_5[imageIndex])
            plt.title("Epsilon 0.5 for " + str(y_test[imageIndex]))

plt.tight_layout()
plt.suptitle("Adversarial Generated Images (0.1, 0.2, 0.5)")
plt.subplots_adjust(top=0.88)
plt.show()
'''
