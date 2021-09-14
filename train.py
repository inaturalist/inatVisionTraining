import os
import time
import numpy as np
import argparse
import yaml

import tensorflow as tf
from tensorflow import keras
AUTOTUNE = tf.data.experimental.AUTOTUNE

from nets import nets


def make_training_callbacks(config):

    def lr_scheduler_fn(epoch):
        return config["INITIAL_LEARNING_RATE"] * \
            tf.math.pow(config["LR_DECAY_FACTOR"], epoch//config["EPOCHS_PER_LR_DECAY"])

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=config["TENSORBOARD_LOG_DIR"],
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq=20,
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata={}
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lr_scheduler_fn,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config["CHECKPOINT_DIR"],
            save_weights_only=True,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1
        ),
        tf.keras.callbacks.experimental.BackupAndRestore(
            backup_dir=config["BACKUP_DIR"],
        ),
    ]

    return callbacks


def main():
    # get command line args
    parser = argparse.ArgumentParser(description="Train an iNat model.")
    parser.add_argument(
        '--config_file',
        required=True,
        help="YAML config file for training."
    )
    args = parser.parse_args()

    # read in config file
    if not os.path.exists(args.config_file):
        print("No config file.")
        return
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    if config["TRAIN_MIXED_PRECISION"]:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    if config["MULTIGPU"]:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    # load train & val datasets from ImageDataGenerator
    if not os.path.exists(config["TRAINING_DATA_DIR"]):
        print("Training data directory doesn't exist.")
        return
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255.,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    train_generator = train_datagen.flow_from_directory(
        config["TRAINING_DATA_DIR"],
        target_size=config["IMAGE_SIZE"],
        class_mode="sparse",
        batch_size=config["BATCH_SIZE"],
        follow_links=True
    )
    if train_generator is None:
        print("No training dataset.")
        return

    if not os.path.exists(config["VAL_DATA_DIR"]):
        print("Validation data file doesn't exist.")
        return
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255
    )
    val_generator = val_datagen.flow_from_directory(
        config["VAL_DATA_DIR"],
        target_size=config["IMAGE_SIZE"],
        class_mode="sparse",
        batch_size=config["BATCH_SIZE"],
        follow_links=True
    )
    if val_generator is None:
        print("No validation dataset")
        return

    with strategy.scope():
        # create optimizer for neural network
        optimizer = keras.optimizers.RMSprop(
            lr=config["INITIAL_LEARNING_RATE"],
            rho=config["RMSPROP_RHO"],
            momentum=config["RMSPROP_MOMENTUM"],
            epsilon=config["RMSPROP_EPSILON"]
        )

        # create neural network
        model = nets.make_neural_network(
            base_arch_name = "xception",
            image_size = config["IMAGE_SIZE"],
            dropout_pct = config["DROPOUT_PCT"],
            n_classes = config["NUM_CLASSES"],
            input_dtype = tf.float16 if config["TRAIN_MIXED_PRECISION"] else tf.float32,
            train_full_network = True
        )
        if model is None:
            print("No model to train.")
            return

        # compile the network for training
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        # setup callbacks
        training_callbacks = make_training_callbacks(config)

        start = time.time()
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=config["NUM_EPOCHS"],
            callbacks=training_callbacks
        )

        end = time.time()
        print(history.history)
        print("time elapsed during fit: {:.1f}".format(end-start))
        model.save(config["FINAL_SAVE_DIR"])


if __name__ == "__main__":
    main()
