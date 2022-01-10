import os
import time
import pandas as pd
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from collections import defaultdict

import json

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

from datasets import inat_dataset


def main():
    # get command line args
    parser = argparse.ArgumentParser(description="Eval an iNat model.")
    parser.add_argument(
        "--config_file", required=True, help="YAML config file for training."
    )
    parser.add_argument(
        "--should_save_results",
        action="store_true",
        help="Save the results from the eval? If yes, --save_file is required.",
    )
    parser.add_argument(
        "--save_file",
        help="Location to save the results of the eval pass. This will be an npz file.",
    )
    args = parser.parse_args()

    # read in config file
    if not os.path.exists(args.config_file):
        parser.error("Could not open config file {}".format(args.config_file))
        return
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # check if save_file and save_results are both flagged
    if args.should_save_results and args.save_file is None:
        parser.error(
            "the --should_save_results argument requires the --save_file argument."
        )
        return

    # load test dataset
    if not os.path.exists(config["TEST_DATA"]):
        print("Training data file doesn't exist.")
        return
    (test_ds, num_test_examples) = inat_dataset.make_dataset(
        config["TEST_DATA"],
        image_size=config["IMAGE_SIZE"],
        batch_size=config["BATCH_SIZE"],
        label_column_name=config["LABEL_COLUMN_NAME"],
        repeat_forever=False,
        augment=False,
    )
    if test_ds is None:
        print("No training dataset.")
        return
    if num_test_examples == 0:
        print("No training examples.")
        return

    # load the model
    model = tf.keras.applications.Xception(weights=None, classes=config["NUM_CLASSES"])
    if model is None:
        assert False, "No model to train."

    model.load_weights(config["FINAL_SAVE_DIR"]).expect_partial()

    # compile for evaluate
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer="rmsprop",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3 accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=10, name="top10 accuracy"),
        ],
    )

    if args.should_save_results:
        all_true = np.array([])
        all_yhat = np.array([])

        for x, y in tqdm(test_ds):
            dense_y = np.argmax(y, axis=1)
            all_true = np.append(all_true, dense_y)
            yhat = np.argmax(model.predict(x), axis=1)
            all_yhat = np.append(all_yhat, yhat)

        np.savez_compressed(args.save_file, all_true=all_true, all_yhat=all_yhat)

    else:
        # the simpler approach, simply eval the test dataset
        model.evaluate(test_ds)


if __name__ == "__main__":
    main()
