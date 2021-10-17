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

import mlflow
import mlflow.tensorflow

from datasets import inat_dataset
from nets import nets
from metrics import make_sparse_parent_accuracy_metric

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
    parser.add_argument(
        '--mlflow_server',
        required=True,
        help="url for the MLFLow service, probably http://localhost:5000"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_server)
    mlflow.tensorflow.autolog()

    with mlflow.start_run():

        # read in config file
        if not os.path.exists(args.config_file):
            print("No config file.")
            return
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
        
        #  log a bunch of config parameters to mlflow
        mlflow.log_param("inat_train_file", config["TRAINING_DATA"])
        
        mlflow.log_param("inat_num_classes", config["NUM_CLASSES"])

        mlflow.log_param("inat_mixed_precision", config["TRAIN_MIXED_PRECISION"])

        mlflow.log_param("inat_multigpu", config["MULTIGPU"])

        mlflow.log_param("inat_batch_size", config["BATCH_SIZE"])

        mlflow.log_param("inat_initial_lr", config["INITIAL_LEARNING_RATE"])
        mlflow.log_param("inat_lr_decay_factor", config["LR_DECAY_FACTOR"])
        mlflow.log_param("inat_epochs_per_lr_decay", config["EPOCHS_PER_LR_DEECAY"])

        mlflow.log_param("inat_pretrained_model", config["PRETRAINED_MODEL"])

        mlflow.log_param("inat_do_label_smooth", config["DO_LABEL_SMOOTH"])
        if config["DO_LABEL_SMOOTH"]:
            mlflow.log_param("inat_label_smooth_mode", config["LABEL_SMOOTH_MODE"])
            mlflow.log_param("label_smooth_pct", config["LABEL_SMOOTH_PCT"])

        mlflow.log_param("inat_model_arch", config["MODEL_NAME"])
        
        mlflow.log_param("inat_dropout_pct", config["DROPOUT_PCT"])

        mlflow.log_param("inat_optimizer", config["OPTIMIZER_NAME"])
        if config["OPTIMIZER_NAME"] == "rmsprop":
            mlflow.log_param("inat_rmsprop_rho", config["RMSPROP_RHO"])
            mlflow.log_param("inat_rmsprop_momentum", config["RMSPROP_MOMENTUM"])
            mlflow.log_param("inat_rmsprop_episilon", config["RMSPROP_EPSILON"])


        if config["TRAIN_MIXED_PRECISION"]:
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)

        if config["MULTIGPU"]:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()

        # load the taxonomy
        if not os.path.exists(config["TAXONOMY_FILE"]):
            print("Taxonomy file doesn't exist.")
            return
        tax = pd.read_csv(config["TAXONOMY_FILE"])
        leaf_tax = tax.dropna(subset=["leaf_class_id"])
        # construct the list of parent class ids
        # we'll use this for our custom parent accuracy metric
        parent_class_ids = [int(x) for x in leaf_tax["parent_taxon_id"]]

        # load train & val datasets
        if not os.path.exists(config["TRAINING_DATA"]):
            print("Training data file doesn't exist.")
            return
        (train_ds, num_train_examples) = inat_dataset.make_dataset(
            config["TRAINING_DATA"],
            image_size=config["IMAGE_SIZE"],
            batch_size=config["BATCH_SIZE"],
            repeat_forever=True,
            augment=True
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
            image_size=config["IMAGE_SIZE"],
            batch_size=config["BATCH_SIZE"],
            repeat_forever=True,
            augment=False
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
                lr=config["INITIAL_LEARNING_RATE"],
                rho=config["RMSPROP_RHO"],
                momentum=config["RMSPROP_MOMENTUM"],
                epsilon=config["RMSPROP_EPSILON"]
            )

            # create neural network
            model = nets.make_neural_network(
                base_arch_name = "xception",
                weights = config["PRETRAINED_MODEL"],
                image_size = config["IMAGE_SIZE"],
                dropout_pct = config["DROPOUT_PCT"],
                n_classes = config["NUM_CLASSES"],
                input_dtype = tf.float16 if config["TRAIN_MIXED_PRECISION"] else tf.float32,
                train_full_network = True
            )

            # load pretrained model
            if False and config["PRETRAINED_MODEL"] != "imagenet" and os.path.exists(config["PRETRAINED_MODEL"]):
                model.load_weights(config["PRETRAINED_MODEL"], by_name=True)


            if model is None:
                print("No model to train.")
                return

            parent_accuracy_metric = make_parent_accuracy_metric(parent_class_ids)
        
            if config["DO_LABEL_SMOOTH"]:
                if config["LABEL_SMOOTH_MODE"] == "flat":
                    # with flat label smoothing we can do it all
                    # in the loss function
                    loss=tf.keras.losses.CategoricalCrossentropy(
                        label_smoothing=config["LABEL_SMOOTH_PCT"]
                    )
                else:
                    # with parent/heirarchical label smoothing
                    # we can't do it in the loss function, we have
                    # to adjust the labels in the dataset
                    print("Unsupported label smoothing mode.")
                    return
            else:
                loss=tf.keras.losses.CategoricalCrossentropy()

            # compile the network for training
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=optimizer,
                metrics=[
                    "accuracy", 
                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3 accuracy"),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=10, name="top10 accuracy"),
                    parent_accuracy_metric,
                ]
            )

            # setup callbacks
            training_callbacks = make_training_callbacks(config)

            STEPS_PER_EPOCH = np.ceil(num_train_examples/config["BATCH_SIZE"])
            VAL_STEPS = np.ceil(num_val_examples/config["BATCH_SIZE"])

            start = time.time()
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                validation_steps=VAL_STEPS,
                epochs=config["NUM_EPOCHS"],
                steps_per_epoch=STEPS_PER_EPOCH,
                callbacks=training_callbacks
            )

            end = time.time()
            print("time elapsed during fit: {:.1f}".format(end-start))
            print(history.history)
            model.save(config["FINAL_SAVE_DIR"])


if __name__ == "__main__":
    main()
