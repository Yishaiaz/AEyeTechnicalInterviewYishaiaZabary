import os
import shutil
import datetime
from copy import copy, deepcopy
import re
import itertools
from typing import *

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Flatten, Input, Dense, Softmax, Conv1D, Conv2DTranspose, BatchNormalization, Dropout, Conv2D, MaxPool2D, concatenate, RNN, GRU, LSTM, ZeroPadding1D, ZeroPadding2D
from tensorflow.keras.activations import softmax, relu, tanh
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy, mean_squared_error, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.utils import plot_model
# from tensorflow.data.Dataset as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import tensorflow_datasets as tfds


def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image

def read_img_and_attach_label(img_path: str):
    """
    read and resize an image, attaching the appropriate label (read from the filename)
    if the filename contains 'left' the label is 0, if it contains 'right' the label is 1, otherwise -1
    :param img_path:
    :return:
    """
    label = [1] if 'right' in img_path.split(os.sep)[-1].lower() else [0] if 'left' in img_path.split(os.sep)[-1].lower() else [-1]
    img = cv.imread(img_path)
    img = cv.resize(img, (1024, 1024))
    img = normalize(img)
    return img, label

train_images_dir_path = 'C:\\Users\\User\\PycharmProjects\\AEyeTechnicalInterview\\Data\\train'
all_train_img_files_paths = np.array(list(map(lambda filename: os.path.join(train_images_dir_path, filename), filter(lambda x: '.jpeg' in x, os.listdir(train_images_dir_path)))))
np.random.shuffle(all_train_img_files_paths)
validation_size = int((len(all_train_img_files_paths)/100) * 20)
train_paths, validation_paths = all_train_img_files_paths[:-validation_size], all_train_img_files_paths[validation_size:]

def my_gen(train=True):
    def data_gen():
        if train:
            for img_full_path in train_paths:
                yield read_img_and_attach_label(img_full_path)
        else:
            for img_full_path in validation_paths:
                yield read_img_and_attach_label(img_full_path)
    return data_gen

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="vertical", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        return inputs, labels


def my_net(input_shape, nfilters_initial):
    inputs = Input(input_shape)

    def convolution_block(block_input, nfilters):
        conv = Conv2D(nfilters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block_input)
        conv = Conv2D(nfilters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
        max_pool = MaxPool2D(pool_size=(2, 2))(conv)

        skip_con = conv
        block_output = max_pool
        return skip_con, block_output

    def dense_classification_block(block_input, n_size):
        dense_l = Dense(n_size, activation='relu')(block_input)
        dense_l = Dense(1, activation='sigmoid')(dense_l)
        # softmax = Softmax()(dense_l)
        return dense_l

    d_1_s, d_1_output = convolution_block(inputs, nfilters=nfilters_initial)
    d_2_s, d_2_output = convolution_block(d_1_output, nfilters=nfilters_initial*2)
    d_3_s, d_3_output = convolution_block(d_2_output, nfilters=nfilters_initial*4)

    d_3_output = tf.keras.layers.Flatten()(
        Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(d_3_output))

    # down_drop = Dropout(0.5)(d_3_output)
    classification_layer = dense_classification_block(d_3_output, n_size=10)

    model = Model(inputs=inputs, outputs=classification_layer)
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=[tf.keras.metrics.BinaryAccuracy()],)
    return model

simple_conv_model = my_net(input_shape=(1024, 1024, 3), nfilters_initial=5)

train_images = tf.data.Dataset.from_generator(my_gen(train=True), output_signature=(tf.TensorSpec(shape=(1024, 1024, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(1,), dtype=tf.int32, name=None)))

validation_images = tf.data.Dataset.from_generator(my_gen(train=False), output_signature=(tf.TensorSpec(shape=(1024, 1024, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(1,), dtype=tf.int32, name=None)))

BUFFER_SIZE = 10
BATCH_SIZE = 1
train_batches = (
    train_images
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .cache()
    # .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

validation_batches = (
    validation_images
    .batch(BATCH_SIZE)
    .cache()
    .repeat(10)
    .prefetch(buffer_size=tf.data.AUTOTUNE))

for im, label in train_batches.take(1):
    print(label[0])
    print(im[0])

print(simple_conv_model.summary())
# train the network as is
simple_conv_model.fit(train_batches, validation_data=validation_batches, epochs=10)

simple_conv_model.save('simple_model_v1')

# predicting the rest of the images:
test_images_dir_path = 'C:\\Users\\User\\PycharmProjects\\AEyeTechnicalInterview\\Data\\test_scrambel'
all_test_img_files_paths = np.array(list(map(lambda filename: os.path.join(train_images_dir_path, filename), filter(lambda x: '.jpeg' in x, os.listdir(test_images_dir_path)))))
all_test_images = np.stack(np.array(list(map(lambda x: read_img_and_attach_label(x)[0], all_test_img_files_paths))))
all_test_prediction = simple_conv_model.predict(all_test_images)
pd.DataFrame({'file_paths': all_test_img_files_paths, 'prediction': list(map(lambda x: 'right' if x else 'left', all_test_prediction))}).to_csv('AEYE_prediction_results.csv')