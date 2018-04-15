#!/usr/bin/env python
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# defaults = {
#     'num_classes': 10,
#     'epochs': 100,
#     'batch_size': 128,
#     'use_cnn': True,
# }


# def main(args):
#
#     img_width, img_height = 150, 150
#
#     train_data_dir = 'biometrics_face/data/train'
#     validation_data_dir = 'biometrics_face/data/validation'
#     nb_train_samples = 7584
#     nb_validation_samples = 400
#     epochs = 10
#     batch_size = 16
#
#     if K.image_data_format() == 'channels_first':
#         input_shape = (3, img_width, img_height)
#     else:
#         input_shape = (img_width, img_height, 3)
#
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), input_shape=input_shape))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(32, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(64))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#
#     model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
#     # this is the augmentation configuration we will use for training
#     train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#
#     # this is the augmentation configuration we will use for testing:
#     # only rescaling
#     test_datagen = ImageDataGenerator(rescale=1. / 255)
#
#     train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
#         batch_size=batch_size, class_mode='binary')
#
#     validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
#         batch_size=batch_size, class_mode='binary')
#
#     model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
#         validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)
#
#     model.save_weights('first_try.h5')



import argparse
import keras
from keras import utils, backend
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Model
from keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy
from PIL import Image
from keras.models import Sequential

defaults = {
    'num_classes': 10,
    'epochs': 100,
    'batch_size': 128,
    'use_cnn': True,
}

# def build_cnn(input_shape, num_classes=10):
#     inputs = Input(shape=input_shape)
#
#     x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
#     x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
#     x = MaxPool2D((2, 2))(x)  # (28, 28) -> (14, 14)
#
#     x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
#     x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
#     x = MaxPool2D((2, 2))(x)  # (14, 14) -> (7, 7)
#
#     x = Flatten()(x)  # 7*7 = 49
#     x = Dense(128, activation='relu')(x)
#     x = Dense(128, activation='relu')(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
#
#     return Model(inputs=inputs, outputs=predictions)







def main(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    img_width, img_height = 224, 224
    batch_size = 16
    epochs = 50
    nb_train_samples = 7589
    nb_validation_samples = 408

    if backend.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    x = keras.layers.Input(input_shape)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    validation_generator = test_datagen.flow_from_directory(
        '/biometrics_face/validation',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    train_generator = train_datagen.flow_from_directory(
        '/biometrics_face/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    model = Sequential()
    model.add(resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Trains either MLP or CNN model on MNIST dataset')
    parser.add_argument(
        '--num-classes',
        help='Number of classes in the classification problem.',
        type=int,
        default=defaults['num_classes'])
    parser.add_argument(
        '--epochs',
        help='Number of epochs to train.',
        type=int,
        default=defaults['epochs'])
    parser.add_argument(
        '--batch-size',
        help='Batch size',
        type=int,
        default=defaults['batch_size'])
    parser.add_argument(
        '--use-cnn',
        help='True if CNN should be used, False for MLP',
        type=bool,
        default=defaults['use_cnn'])

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
