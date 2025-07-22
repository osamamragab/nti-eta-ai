import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub


LABELS = ["PNEUMONIA", "NORMAL"]
IMG_SIZE = 150


def get_training_data(data_dir: str):
    images = []
    labels = []
    for label in LABELS:
        path = os.path.join(data_dir, label)
        class_num = LABELS.index(label)
        for img in os.listdir(path):
            if not img.endswith(".jpeg"):
                continue
            try:
                img_arr = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
                if img_arr is None:
                    print(f"Failed to load image: {img}")
                    continue
                resized_arr = cv.resize(img_arr, (IMG_SIZE, IMG_SIZE))
                images.append(resized_arr)
                labels.append(class_num)
            except Exception as e:
                print(e)
    return np.array(images), np.array(labels)


def main():
    dataset_root_path = kagglehub.dataset_download(
        "paultimothymooney/chest-xray-pneumonia"
    )
    dataset_path = os.path.join(dataset_root_path, "chest_xray", "chest_xray")
    print("dataset path:", dataset_path)

    train_data, train_labels = get_training_data(os.path.join(dataset_path, "train"))
    test_data, test_labels = get_training_data(os.path.join(dataset_path, "test"))
    val_data, val_labels = get_training_data(os.path.join(dataset_path, "val"))

    train_cl = []
    for i in train_labels:
        train_cl.append("Pneumonia" if i == 0 else "Normal")
    sns.set_style("darkgrid")
    sns.countplot(train_cl)

    plt.figure(figsize=(5, 5))
    plt.imshow(train_data[0], cmap="gray")
    plt.title(LABELS[train_labels[0]])
    plt.figure(figsize=(5, 5))
    plt.imshow(train_data[-1], cmap="gray")
    plt.title(LABELS[train_labels[-1]])

    x_train = np.array(train_data) / 255
    x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array(train_labels)

    x_test = np.array(test_data) / 255
    x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = np.array(test_labels)

    x_val = np.array(val_data) / 255
    x_val = x_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_val = np.array(val_labels)

    img = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
    )
    img.fit(x_train)

    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                strides=1,
                padding="same",
                activation="relu",
                input_shape=(150, 150, 1),
            ),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
            Dropout(0.1),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"),
            Dropout(0.2),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"),
            Dropout(0.2),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Flatten(),
            Dense(units=128, activation="relu"),
            Dropout(0.2),
            Dense(units=1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_accuracy", patience=2, verbose=1, factor=0.3, min_lr=0.000001
    )

    history = model.fit(
        img.flow(x_train, y_train, batch_size=32),
        epochs=12,
        validation_data=img.flow(x_val, y_val),
        callbacks=[learning_rate_reduction],
    )

    print("Loss of the model is - ", model.evaluate(x_test, y_test)[0])
    print("Accuracy of the model is - ", model.evaluate(x_test, y_test)[1] * 100, "%")


if __name__ == "__main__":
    main()
