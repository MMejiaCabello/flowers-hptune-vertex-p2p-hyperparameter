import tensorflow as tf
import numpy as np
import os
import hypertune
import argparse

## Replace {your-gcs-bucket} !!
BUCKET_ROOT='/gcs/prototype-to-production-405205-bucket'

# Define variables
NUM_CLASSES = 5
EPOCHS=10
BATCH_SIZE = 32

IMG_HEIGHT = 180
IMG_WIDTH = 180

DATA_DIR = f'{BUCKET_ROOT}/flower_photos'

def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        required=True,
        type=float,
        help='learning rate')
    parser.add_argument(
        '--momentum',
        required=True,
        type=float,
        help='SGD momentum value')
    parser.add_argument(
        '--num_units',
        required=True,
        type=int,
        help='number of units in last hidden layer')
    args = parser.parse_args()
    return args

def create_datasets(data_dir, batch_size):
    '''Creates train and validation datasets.'''

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size)

    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset

def create_model(num_units, learning_rate, momentum):
    '''Creates model.'''

    model = tf.keras.Sequential([
        tf.keras.layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_units, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def main():
    args = get_args()
    train_dataset, validation_dataset = create_datasets(DATA_DIR, BATCH_SIZE)
    model = create_model(args.num_units, args.learning_rate, args.momentum)
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

    # DEFINE METRIC
    hp_metric = history.history['val_accuracy'][-1]

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=hp_metric,
        global_step=EPOCHS)

if __name__ == "__main__":
    main()
