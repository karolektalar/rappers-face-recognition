import argparse
import keras
from keras import backend
from keras.layers import Dense, Flatten
from keras.applications import resnet50
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import Sequential

defaults = {
    'num_classes': 10,
    'epochs': 100,
    'batch_size': 128,
    'use_cnn': True,
}


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
