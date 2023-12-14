"""
The program uses a neural network to predict the class of tree:
0: baobab
1: brzoza
2: tuja

Authors:
    Oliwier Kossak s22018
    Daniel Klimowski S18504

Preparing the environment:
    libraries to install: tensorflow, numpy, matplotlib
    to install  use command: "pip install <name of library >"
"""

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import math
import shutil
import numpy as np


def split_size_train_validation_test(size, training_size=0.7, validation_size=0.2):
    """
    Function that return size of validation, training, and testing set.

    Parameters:
        size (int): Number of images.
        training_size (float): Size of training set.
        validation_size (float): Size of validation set.
    Returns:
        int: Size of training, validation and test set.
    """
    training = math.floor(training_size * size)
    validation = math.floor(validation_size * size)
    test = size - training - validation

    return training, validation, test


def create_directories_training_validation_test(classes):
    """Function that create directory with directories training, validation and test."""

    path = "data"
    directories = ['training', 'validation', 'test']
    if not os.path.exists(path):
        for dir in directories:
            for object_class in classes:
                os.makedirs(f"data/{dir}/{object_class}")


def copy_images_training_validation_test_directories(object_class, training_size, validation_size):
    """Function that copies images from an image set and divides them into directories for training, validation,
       and testing the dataset."""

    images_path = f'images/{object_class}/'
    images_number = len(os.listdir(images_path))
    images = []

    for i in os.listdir(images_path):
        images.append(i)

    for image_id in range(images_number):
        if image_id < training_size:
            str_path = f'images/{object_class}/{images[image_id]}'
            dsc_path = f'data/training/{object_class}'
            shutil.copy(str_path, dsc_path)
        elif image_id < (training_size + validation_size):
            str_path = f'images/{object_class}/{images[image_id]}'
            dsc_path = f'data/validation/{object_class}'
            shutil.copy(str_path, dsc_path)
        else:
            str_path = f'images/{object_class}/{images[image_id]}'
            dsc_path = f'data/test/{object_class}'
            shutil.copy(str_path, dsc_path)

"""tree classes"""
object_classes = ['baobab', 'brzoza', 'tuja']

"""Creates directories for training, validation and test set."""
create_directories_training_validation_test(object_classes)

"""Copies images to directories for training, validation and test set."""
for object_class in object_classes:
    path = f'images/{object_class}/'
    size = len(os.listdir(path))
    training, validation, test = split_size_train_validation_test(size)
    copy_images_training_validation_test_directories(object_class, training, validation)

"""Artificially increasing the size of the dataset by introducing various transformations to the original images."""
train_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,  # random rotation of image
    rescale=1. / 255.,
    width_shift_range=0.15,  # vertical image transformation
    height_shift_range=0.15,  # horizontal image transformation
    shear_range=0.15,  # random cropping range
    zoom_range=0.15,  # random zoom range
    horizontal_flip=True,  # random reflection of half of the image in the horizontal plane
    fill_mode='nearest',  # filling the newly created pixels
)

"""Normalizes the pixel values of the validation data to the range 0 to 1 by dividing each pixel by 255"""
validation_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)

"""Creates a training data generator."""
train_generator = train_data.flow_from_directory(directory='data/training/',
                                                 batch_size=4,
                                                 target_size=(150, 150),
                                                 class_mode='categorical')

"""Creates a validation data generator."""
valid_generator = validation_data.flow_from_directory(directory='data/validation/',
                                                      batch_size=4,
                                                      target_size=(150, 150),
                                                      class_mode='categorical')

model_path = 'model'
history = None
model = None

"""Creates model if not exists"""
if not os.path.exists(model_path):
    """Neural network model"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    batch_size = 4

    steps_per_epoch = len(os.listdir('data/training/baobab/')) // batch_size
    valid_steps = len(os.listdir('data/validation/baobab/')) // batch_size

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=40,

                                  validation_data=valid_generator,
                                  validation_steps=valid_steps)
    os.makedirs(model_path)
    model.save(f"{model_path}/CNN.model")
    plt.plot(history.epoch, history.history['accuracy'])
    plt.show()

else:
    model = tf.keras.models.load_model(f'{model_path}/CNN.model')


"""Normalizes the pixel values of the test data to the range 0 to 1 by dividing each pixel by 255"""
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)

"""Creates a test data generator."""
test_generator = validation_data.flow_from_directory(directory='data/test/',
                                                      batch_size=4,
                                                      target_size=(150, 150),
                                                      class_mode='categorical',
                                                     shuffle=False)

"""Model predictions of samples in the test set"""
y_prob = model.predict_generator(test_generator, test_generator.samples)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_generator.classes


"""Model evaluation,"""
correct_prediction = 0

for i in range(len(y_pred)):
    if y_pred[i] == y_true[i]:
        correct_prediction += 1

accuracy = round(correct_prediction/len(y_pred), 2)
print(f"{accuracy * 100}%")
