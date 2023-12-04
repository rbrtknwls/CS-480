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
    image = np.expand_dims(image, axis=0)
    image = tf.convert_to_tensor(image)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = lossObject(label, prediction)

    gradient = tape.gradient(loss, image)
    return np.array(gradient[0])


trainedModel1 = tf.keras.models.load_model('../Midterm/MINSTtrained.keras')

probModel = tf.keras.Sequential([
    trainedModel1,
    layers.Softmax()
])

trainedModel2 = tf.keras.models.load_model('../Midterm/MINSTtrained.keras')

probModel1FGSM = tf.keras.Sequential([
    trainedModel2,
    layers.Softmax()
])

trainedModel3 = tf.keras.models.load_model('../Midterm/MINSTtrained.keras')

probModel2FGSM = tf.keras.Sequential([
    trainedModel3,
    layers.Softmax()
])

trainedModel4 = tf.keras.models.load_model('../Midterm/MINSTtrained.keras')

probModel5FGSM = tf.keras.Sequential([
    trainedModel4,
    layers.Softmax()
])

trainedModel5 = tf.keras.models.load_model('../Midterm/MINSTtrained.keras')

probModel1PGD = tf.keras.Sequential([
    trainedModel5,
    layers.Softmax()
])

trainedModel6 = tf.keras.models.load_model('../Midterm/MINSTtrained.keras')

probModel2PGD = tf.keras.Sequential([
    trainedModel6,
    layers.Softmax()
])

trainedModel7 = tf.keras.models.load_model('../Midterm/MINSTtrained.keras')

probModel5PGD = tf.keras.Sequential([
    trainedModel7,
    layers.Softmax()
])

probModel.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

HYPERPARAM = 0.5

# ======================== Compile Tests ========================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

Normal = []

FGSM_1 = []
FGSM_2 = []
FGSM_5 = []

PGD_1 = []
PGD_2 = []
PGD_5 = []

for i in range(0, len(x_test)):
    norm = normalize_img(x_test[i])
    sparseLabel = []
    print(i)

    for x in range(0, 10):
        if x == y_test[i]:
            sparseLabel.append(1.0)
        else:
            sparseLabel.append(0)

    sparseLabel = tf.convert_to_tensor(np.expand_dims(sparseLabel, axis=0))

    Normal.append(norm)

    delta = generate_adversarial_image(norm, sparseLabel, probModel)
    signedDelta = np.sign(delta)

    # FGSM

    image_1 = np.clip(np.add(norm, signedDelta * 0.1), 0, 1)
    image_2 = np.clip(np.add(norm, signedDelta * 0.2), 0, 1)
    image_3 = np.clip(np.add(norm, signedDelta * 0.5), 0, 1)

    FGSM_1.append(image_1)
    FGSM_2.append(image_2)
    FGSM_5.append(image_3)

    # PGD

    image_1 = np.add(norm, np.clip(signedDelta*HYPERPARAM, -0.1, 0.1))
    image_2 = np.add(norm, np.clip(signedDelta*HYPERPARAM, -0.2, 0.2))
    image_3 = np.add(norm, np.clip(signedDelta*HYPERPARAM, -0.5, 0.5))

    delta_1 = np.sign(generate_adversarial_image(image_1, sparseLabel, probModel))
    delta_2 = np.sign(generate_adversarial_image(image_2, sparseLabel, probModel))
    delta_3 = np.sign(generate_adversarial_image(image_3, sparseLabel, probModel))

    image_1 = np.add(norm, np.clip(np.add(signedDelta * HYPERPARAM, HYPERPARAM * delta_1), -0.1, 0.1))
    image_2 = np.add(norm, np.clip(np.add(signedDelta * HYPERPARAM, HYPERPARAM * delta_2), -0.2, 0.2))
    image_3 = np.add(norm, np.clip(np.add(signedDelta * HYPERPARAM, HYPERPARAM * delta_3), -0.5, 0.5))

    PGD_1.append(np.clip(image_1, 0, 1))
    PGD_2.append(np.clip(image_2, 0, 1))
    PGD_5.append(np.clip(image_3, 0, 1))

Normal = np.array(Normal)

FGSM_1 = np.array(FGSM_1)
FGSM_2 = np.array(FGSM_2)
FGSM_5 = np.array(FGSM_5)

PGD_1 = np.array(PGD_1)
PGD_2 = np.array(PGD_2)
PGD_5 = np.array(PGD_5)

for epsilonChoice in range(0, 3):
    for imageIndex in range(0,5):
        plt.subplot(3, 5, epsilonChoice*5+imageIndex+1)
        if epsilonChoice == 0:
            plt.imshow(FGSM_1[imageIndex], cmap='gray')
            predicted = np.argmax(probModel.predict(np.expand_dims(FGSM_1[imageIndex], axis=0)))

            if predicted != y_test[imageIndex]:
                plt.title("Predicted:" + str(predicted), color='red')
            else:
                plt.title("Predicted:" + str(predicted), color='green')

        elif epsilonChoice == 1:
            plt.imshow(FGSM_2[imageIndex], cmap='gray')
            predicted = np.argmax(probModel.predict(np.expand_dims(FGSM_2[imageIndex], axis=0)))

            if predicted != y_test[imageIndex]:
                plt.title("Predicted:" + str(predicted), color='red')
            else:
                plt.title("Predicted:" + str(predicted), color='green')

        else:
            plt.imshow(FGSM_5[imageIndex], cmap='gray')
            predicted = np.argmax(probModel.predict(np.expand_dims(FGSM_5[imageIndex], axis=0)))

            if predicted != y_test[imageIndex]:
                plt.title("Predicted:" + str(predicted), color='red')
            else:
                plt.title("Predicted:" + str(predicted), color='green')

plt.tight_layout()
params = {"text.color": "black"}
plt.rcParams.update(params)
plt.suptitle("FGSM Adversarial Generated Images (0.1, 0.2, 0.5)")
plt.subplots_adjust(top=0.88)
plt.show()

for epsilonChoice in range(0, 3):
    for imageIndex in range(0,5):
        plt.subplot(3, 5, epsilonChoice*5+imageIndex+1)
        if epsilonChoice == 0:
            plt.imshow(PGD_1[imageIndex], cmap='gray')

            predicted = np.argmax(probModel.predict(np.expand_dims(PGD_1[imageIndex], axis=0)))

            if predicted != y_test[imageIndex]:
                plt.title("Predicted:" + str(predicted), color='red')
            else:
                plt.title("Predicted:" + str(predicted), color='green')

        elif epsilonChoice == 1:
            plt.imshow(PGD_2[imageIndex], cmap='gray')

            predicted = np.argmax(probModel.predict(np.expand_dims(PGD_2[imageIndex], axis=0)))

            if predicted != y_test[imageIndex]:
                plt.title("Predicted:" + str(predicted), color='red')
            else:
                plt.title("Predicted:" + str(predicted), color='green')

        else:
            plt.imshow(PGD_5[imageIndex], cmap='gray')
            predicted = np.argmax(probModel.predict(np.expand_dims(PGD_5[imageIndex], axis=0)))

            if predicted != y_test[imageIndex]:
                plt.title("Predicted:" + str(predicted), color='red')
            else:
                plt.title("Predicted:" + str(predicted), color='green')

plt.tight_layout()
params = {"text.color": "black"}
plt.rcParams.update(params)
plt.suptitle("PGD Adversarial Generated Images (0.1, 0.2, 0.5)")
plt.subplots_adjust(top=0.88)
plt.show()


# ======================== Train the Models ========================

probModel1FGSM.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

probModel2FGSM.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

probModel5FGSM.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

probModel1PGD.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

probModel2PGD.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

probModel5PGD.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

FGSM_1_Train = []
FGSM_2_Train = []
FGSM_5_Train = []

PGD_1_Train = []
PGD_2_Train = []
PGD_5_Train = []

for i in range(0, len(x_train)):
    norm = normalize_img(x_train[i])
    sparseLabel = []
    print(i)
    for x in range(0, 10):
        if x == y_train[i]:
            sparseLabel.append(1.0)
        else:
            sparseLabel.append(0)

    sparseLabel = tf.convert_to_tensor(np.expand_dims(sparseLabel, axis=0))

    delta = generate_adversarial_image(norm, sparseLabel, probModel)
    signedDelta = np.sign(delta)

    # FGSM

    image_1 = np.clip(np.add(norm, signedDelta * 0.1), 0, 1)
    image_2 = np.clip(np.add(norm, signedDelta * 0.2), 0, 1)
    image_3 = np.clip(np.add(norm, signedDelta * 0.5), 0, 1)

    FGSM_1_Train.append(image_1)
    FGSM_2_Train.append(image_2)
    FGSM_5_Train.append(image_3)

    # PGD

    image_1 = np.add(norm, np.clip(signedDelta*HYPERPARAM, -0.1, 0.1))
    image_2 = np.add(norm, np.clip(signedDelta*HYPERPARAM, -0.1, 0.1))
    image_3 = np.add(norm, np.clip(signedDelta*HYPERPARAM, -0.1, 0.1))

    delta_1 = np.sign(generate_adversarial_image(image_1, sparseLabel, probModel))
    delta_2 = np.sign(generate_adversarial_image(image_2, sparseLabel, probModel))
    delta_3 = np.sign(generate_adversarial_image(image_3, sparseLabel, probModel))

    image_1 = np.add(norm, np.clip(np.add(signedDelta * HYPERPARAM, HYPERPARAM * delta_1), -0.1, 0.1))
    image_2 = np.add(norm, np.clip(np.add(signedDelta * HYPERPARAM, HYPERPARAM * delta_2), -0.2, 0.2))
    image_3 = np.add(norm, np.clip(np.add(signedDelta * HYPERPARAM, HYPERPARAM * delta_3), -0.5, 0.5))

    PGD_1_Train.append(np.clip(image_1, 0, 1))
    PGD_2_Train.append(np.clip(image_2, 0, 1))
    PGD_5_Train.append(np.clip(image_3, 0, 1))



FGSM_1_Train = np.array(FGSM_1_Train)
FGSM_2_Train = np.array(FGSM_2_Train)
FGSM_5_Train = np.array(FGSM_5_Train)

PGD_1_Train = np.array(PGD_1_Train)
PGD_2_Train = np.array(PGD_2_Train)
PGD_5_Train = np.array(PGD_5_Train)

probModel1FGSM.fit(FGSM_1_Train, y_train)
probModel2FGSM.fit(FGSM_2_Train, y_train)
probModel5FGSM.fit(FGSM_5_Train, y_train)

probModel1PGD.fit(PGD_1_Train, y_train)
probModel2PGD.fit(PGD_2_Train, y_train)
probModel5PGD.fit(PGD_5_Train, y_train)


# ======================== Test the Models ========================

print("EPSI 1")

probModel.evaluate(Normal, y_test)
probModel.evaluate(FGSM_1, y_test)
probModel.evaluate(PGD_1, y_test)

probModel2FGSM.evaluate(Normal, y_test)
probModel2FGSM.evaluate(FGSM_1, y_test)
probModel2FGSM.evaluate(PGD_1, y_test)

probModel2PGD.evaluate(Normal, y_test)
probModel2PGD.evaluate(FGSM_1, y_test)
probModel2PGD.evaluate(PGD_1, y_test)

print("EPSI 2")

probModel.evaluate(Normal, y_test)
probModel.evaluate(FGSM_2, y_test)
probModel.evaluate(PGD_2, y_test)

probModel2FGSM.evaluate(Normal, y_test)
probModel2FGSM.evaluate(FGSM_2, y_test)
probModel2FGSM.evaluate(PGD_2, y_test)

probModel2PGD.evaluate(Normal, y_test)
probModel2PGD.evaluate(FGSM_2, y_test)
probModel2PGD.evaluate(PGD_2, y_test)

print("EPSI 3")

probModel.evaluate(Normal, y_test)
probModel.evaluate(FGSM_5, y_test)
probModel.evaluate(PGD_5, y_test)

probModel2FGSM.evaluate(Normal, y_test)
probModel2FGSM.evaluate(FGSM_5, y_test)
probModel2FGSM.evaluate(PGD_5, y_test)

probModel2PGD.evaluate(Normal, y_test)
probModel2PGD.evaluate(FGSM_5, y_test)
probModel2PGD.evaluate(PGD_5, y_test)

