import os
import time
import pandas as pd
import numpy as np
import argparse
import yaml

import json

import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.experimental.AUTOTUNE

from datasets import inat_dataset
from nets import nets


def make_training_callbacks(config):
    def lr_scheduler_fn(epoch):
        return config["INITIAL_LEARNING_RATE"] * tf.math.pow(
            config["LR_DECAY_FACTOR"], epoch // config["EPOCHS_PER_LR_DECAY"]
        )

    checkpoint_file_name = "checkpoint-{epoch:02d}-{val_accuracy:.2f}"
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=config["TENSORBOARD_LOG_DIR"],
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq=20,
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata={},
            write_steps_per_second=True,
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler_fn, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config["CHECKPOINT_DIR"], checkpoint_file_name),
            save_weights_only=True,
            save_best_only=False,
            monitor="val_accuracy",
            verbose=1,
        ),
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=config["BACKUP_DIR"],
        ),
    ]

    return callbacks


def main():
    # get command line args
    parser = argparse.ArgumentParser(description="Train an iNat model.")
    parser.add_argument(
        "--config_file", required=True, help="YAML config file for training."
    )
    args = parser.parse_args()

    # read in config file
    if not os.path.exists(args.config_file):
        print("No config file.")
        return
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    if config["TRAIN_MIXED_PRECISION"]:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    if config["MULTIGPU"]:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    # load train & val datasets
    if not os.path.exists(config["TRAINING_DATA"]):
        print("Training data file doesn't exist.")
        return
    (train_ds, num_train_examples) = inat_dataset.make_dataset(
        config["TRAINING_DATA"],
        label_column_name=config["LABEL_COLUMN_NAME"],
        image_size=config["IMAGE_SIZE"],
        batch_size=config["BATCH_SIZE"],
        shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
        repeat_forever=True,
        augment=True,
    )
    if train_ds is None:
        print("No training dataset.")
        return
    if num_train_examples == 0:
        print("No training examples.")
        return

    if not os.path.exists(config["VAL_DATA"]):
        print("Validation data file doesn't exist.")
        return
    (val_ds, num_val_examples) = inat_dataset.make_dataset(
        config["VAL_DATA"],
        label_column_name=config["LABEL_COLUMN_NAME"],
        image_size=config["IMAGE_SIZE"],
        batch_size=config["BATCH_SIZE"],
        shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
        repeat_forever=True,
        augment=False,
    )
    if val_ds is None:
        print("No val dataset.")
        return
    if num_val_examples == 0:
        print("No val examples.")
        return

    with strategy.scope():
        # create optimizer for neural network
        optimizer = keras.optimizers.RMSprop(
            learning_rate=config["INITIAL_LEARNING_RATE"],
            rho=config["RMSPROP_RHO"],
            momentum=config["RMSPROP_MOMENTUM"],
            epsilon=config["RMSPROP_EPSILON"],
        )

        # create neural network
        model = nets.make_neural_network(
            base_arch_name="xception",
            weights=config["PRETRAINED_MODEL"],
            image_size=config["IMAGE_SIZE"],
            n_classes=config["NUM_CLASSES"],
            input_dtype=tf.float16 if config["TRAIN_MIXED_PRECISION"] else tf.float32,
            train_full_network=config["TRAIN_FULL_MODEL"],
            ckpt=config["CHECKPOINT"] if "CHECKPOINT" in config else None,
        )

        if model is None:
            assert False, "No model to train."

        if config["DO_LABEL_SMOOTH"]:
            if config["LABEL_SMOOTH_MODE"] == "flat":
                # with flat label smoothing we can do it all
                # in the loss function
                loss = tf.keras.losses.CategoricalCrossentropy(
                    label_smoothing=config["LABEL_SMOOTH_PCT"]
                )
            else:
                # with parent/heirarchical label smoothing
                # we can't do it in the loss function, we have
                # to adjust the labels in the dataset
                assert False, "Unsupported label smoothing mode."
        else:
            loss = tf.keras.losses.CategoricalCrossentropy()

        # compile the network for training
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=[
                "accuracy",
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3 accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=10, name="top10 accuracy"),
            ],
        )

        # setup callbacks
        training_callbacks = make_training_callbacks(config)

        # training & val step counts
        STEPS_PER_EPOCH = np.ceil(num_train_examples / config["BATCH_SIZE"])
        VAL_IMAGE_COUNT = (
            config["VALIDATION_PASS_SIZE"]
            if config["VALIDATION_PASS_SIZE"] is not None
            else num_val_examples
        )
        VAL_STEPS = np.ceil(VAL_IMAGE_COUNT / config["BATCH_SIZE"])
        print(
            "{} val steps for {} val pass images of {} total val images.".format(
                VAL_STEPS, VAL_IMAGE_COUNT, num_val_examples
            )
        )

        start = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            validation_steps=VAL_STEPS,
            epochs=config["NUM_EPOCHS"],
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=training_callbacks,
        )

        end = time.time()
        print("time elapsed during fit: {:.1f}".format(end - start))
        print(history.history)
        model.save(config["FINAL_SAVE_DIR"])


if __name__ == "__main__":
    main()
