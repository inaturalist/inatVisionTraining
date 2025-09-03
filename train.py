import os
import time
import pandas as pd
import numpy as np
import argparse
import yaml
import shutil
from pathlib import Path

import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
import wandb
from wandb.integration.keras import WandbMetricsLogger

AUTOTUNE = tf.data.experimental.AUTOTUNE

from datasets import inat_dataset
from nets import nets

class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"learning rate is {lr:.6g}")
        wandb.log({"lr": lr}, commit=False)

class WarmupThenConstant(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, name="warmup_then_constant"):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.name = name


    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = tf.cast(self.warmup_steps, tf.float32)
        base = tf.cast(self.base_lr, tf.float32)
        return tf.cond(
            step < warm, 
            lambda: base * (step / warm), 
            lambda: base
        )

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "name": self.name
        }


def make_training_callbacks(config):
    def lr_scheduler_fn(epoch):
        lr_value = config["INITIAL_LEARNING_RATE"] * (
            config["LR_DECAY_FACTOR"] ** (epoch // config["EPOCHS_PER_LR_DECAY"])
        )
        return float(lr_value)

    checkpoint_file_name = "checkpoint-{epoch:02d}-{val_accuracy:.2f}.weights.h5"
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
        #tf.keras.callbacks.LearningRateScheduler(lr_scheduler_fn, verbose=1),
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
        WandbMetricsLogger(log_freq=config["WANDB_LOG_FREQ"]),
        LRLogger(),
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

    wandb.init(
        project=config["WANDB_PROJECT"],
        config=config
    )

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
        augment=False,
        class_balance=50
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
        lr_schedule = WarmupThenConstant(
            base_lr=config["INITIAL_LEARNING_RATE"],
            warmup_steps=1_000
        )
        optimizer = keras.optimizers.RMSprop(
            learning_rate=lr_schedule,
            rho=config["RMSPROP_RHO"],
            momentum=config["RMSPROP_MOMENTUM"],
            epsilon=config["RMSPROP_EPSILON"],
        )
        
        # loss scale optimizer to prevent numeric underflow
        if config["TRAIN_MIXED_PRECISION"]:
            pass
            # this should be built into the mp policy now?
            #optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        # create neural network
        model = nets.make_neural_network(
            base_arch_name=config["MODEL_NAME"],
            weights=config["PRETRAINED_MODEL"],
            image_size=config["IMAGE_SIZE"],
            n_classes=config["NUM_CLASSES"],
            input_dtype=tf.float32,
            #input_dtype=tf.float16 if config["TRAIN_MIXED_PRECISION"] else tf.float32,
            train_full_network=config["TRAIN_FULL_MODEL"],
            ckpt=config["CHECKPOINT"] if "CHECKPOINT" in config else None,
            factorize=config["FACTORIZE_FINAL_LAYER"] if "FACTORIZE_FINAL_LAYER" in config else False,
            fact_rank=config["FACT_RANK"] if "FACT_RANK" in config else None,
            activation=config["ACTIVATION"] if "ACTIVATION" in config else None,
        )
        
        if model is None:
            assert False, "No model to train."

        if config["ACTIVATION"] is None:
            from_logits = True
        else:
            from_logits = False

        print(f"from logits is {from_logits}")

        if config["DO_LABEL_SMOOTH"]:
            if config["LABEL_SMOOTH_MODE"] == "flat":
                # with flat label smoothing we can do it all
                # in the loss function
                loss = tf.keras.losses.CategoricalCrossentropy(
                    label_smoothing=config["LABEL_SMOOTH_PCT"],
                    from_logits=from_logits
                )
            else:
                # with parent/heirarchical label smoothing
                # we can't do it in the loss function, we have
                # to adjust the labels in the dataset
                assert False, "Unsupported label smoothing mode."
        else:
            print("sparse cross entropy loss, from logits")
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

        # compile the network for training
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3 accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10 accuracy"),
            ],
        )

        # setup callbacks
        training_callbacks = make_training_callbacks(config)

        # training & val step counts
        STEPS_PER_EPOCH = int(np.ceil(num_train_examples / config["BATCH_SIZE"]))
        VAL_IMAGE_COUNT = (
            config["VALIDATION_PASS_SIZE"]
            if config["VALIDATION_PASS_SIZE"] is not None
            else num_val_examples
        )
        VAL_STEPS = int(np.ceil(VAL_IMAGE_COUNT / config["BATCH_SIZE"]))
        print(
            "{} val steps for {} val pass images of {} total val images.".format(
                VAL_STEPS, VAL_IMAGE_COUNT, num_val_examples
            )
        )

        phases = [
            {
                "name": "phase 1",
                "desc": "240x240, noaug, balanced, head only",
                "batch_size": 256,
                "image_size": (240, 240),
                "augment": "none",
                "class_balance": 50,
                "epochs": 1,
                "mode": "head_only",
                "target_lr": 0.03,
                "warmup_steps": 1_000,
                "clipnorm": None,
            },
            {
                "name": "phase 2",
                "desc": "300x300, noaug, balanced, head only",
                "batch_size": 256,
                "image_size": (300, 300),
                "augment": "none",
                "class_balance": 50,
                "epochs": 1,
                "mode": "head_only",
                "target_lr": 0.03,
                "warmup_steps": 2_000,
                "clipnorm": None,
            },
            {
                "name": "phase 3",
                "desc": "300x300, light aug, balanced, last 1/3",
                "batch_size": 256,
                "image_size": (300, 300),
                "augment": "light",
                "class_balance": 50,
                "epochs": 5,
                "mode": "last_third",
                "target_lr": 1e-4,
                "warmup_steps": 3_000,
                "clipnorm": None,
            },
            {
                "name": "phase 4",
                "desc": "300x300, full aug, unbalanced, full model",
                "batch_size": 256,
                "image_size": (300, 300),
                "augment": "full",
                "class_balance": None,
                "epochs": 10,
                "mode": "full",
                "target_lr": 1e-4,
                "warmup_steps": 4_000,
                "clipnorm": 1.0,
            },
        ]
        start = time.time()

        cumulative_epoch = 0 
        for phase in phases:
            print(f"{phase['name']}")
            print(f"{phase}")
            print("make ds")
            (train_ds, num_train_examples) = inat_dataset.make_dataset(
                config["TRAINING_DATA"],
                label_column_name=config["LABEL_COLUMN_NAME"],
                image_size=phase["image_size"],
                batch_size=phase["batch_size"],
                shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
                repeat_forever=True,
                augment=phase["augment"],
                class_balance=phase["class_balance"],
            )
            print("calculating steps per epoch")
            STEPS_PER_EPOCH = int(np.ceil(num_train_examples / phase["batch_size"]))

            #print("dry run, tiny epochs")
            #STEPS_PER_EPOCH = 3
            #VAL_STEPS = 1

            print("adjusting lr")
            lr_schedule = WarmupThenConstant(
                phase["target_lr"],
                phase["warmup_steps"],
            )

            print("setting trainable layers")
            base_model = model.layers[1]
            if phase["mode"] == "head_only":
                base_model.trainable = False
            elif phase["mode"] == "last_third":
                cut = int(len(base_model.layers) * (2/3))
                for i, layer in enumerate(base_model.layers):
                    if isinstance(layer, BatchNormalization):
                        layer.trainable = False
                    else:
                        layer.trainable = (i >= cut)
            elif phase["mode"] == "full":
                for layer in base_model.layers:
                    if isinstance(layer, BatchNormalization):
                        layer.trainable = False
                    else:
                        layer.trainable = True
            else:
                assert False, "unknown phase mode"

            print("recompiling model")
            optimizer = keras.optimizers.RMSprop(
                learning_rate=lr_schedule,
                rho=config["RMSPROP_RHO"],
                momentum=config["RMSPROP_MOMENTUM"],
                epsilon=config["RMSPROP_EPSILON"],
                clipnorm=phase["clipnorm"],
            )

            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3 accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10 accuracy"),
            ]
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            print("calculating epoch counters")
            epochs = cumulative_epoch + phase["epochs"]
            initial_epoch = cumulative_epoch

            print("remaking training callbacks")
            training_callbacks = make_training_callbacks(config)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                validation_steps=VAL_STEPS,
                epochs=epochs,
                initial_epoch=initial_epoch,
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=training_callbacks,
            )

            cumulative_epoch = epochs

            print("rotating the backup dirs")
            new_backup_dir = config["BACKUP_DIR"] + "-" + phase["name"]
            src = Path(config["BACKUP_DIR"])
            dst = Path(new_backup_dir)
            if src.exists():
                os.replace(src, dst)


        end = time.time()
        print("time elapsed during fit: {:.1f}".format(end - start))
        print(history.history)
        model.save(config["FINAL_SAVE_PATH"])

    wandb.finish()

if __name__ == "__main__":
    main()
