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

from datasets.inat_dataset import process_row


def load_test_ds(dataset_json_path, label_column_name, image_size):
    """
    don't use the standard inat dataset loader.
    we load our test ds differently because we need insight into the imageids
    for post eval analysis.
    """
    df = pd.read_json(dataset_json_path)
    num_classes = len(df[label_column_name].unique())
    imageids = list(df["id"])

    ds = tf.data.Dataset.from_tensor_slices((df["filename"], df[label_column_name]))

    process_partial = partial(process_row, num_classes=num_classes)
    ds = ds.map(process_partial, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.image.resize(x, image_size), y))

    return (ds, num_examples, imageids)


def main():
    # get command line args
    parser = argparse.ArgumentParser(description="Eval an iNat model.")
    parser.add_argument(
        "--config_file", required=True, help="YAML config file for training."
    )
    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="yes if use checkpoint, no if use final export",
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
    (test_ds, num_test_examples, imageids) = load_test_ds(
        config["TEST_DATA"],
        label_column_name=config["LABEL_COLUMN_NAME"],
        image_size=config["IMAGE_SIZE"],
    )
    assert test_ds is not None, "No training dataset"
    assert num_test_examples != 0, "num test examples is zero"
    assert imageids is not None, "No imageids"

    # load the model
    model = tf.keras.applications.Xception(weights=None, classes=config["NUM_CLASSES"])
    assert model is not None, "No model to train"

    # load the weights
    weights = (
        config["CHECKPOINT_DIR"] if args.use_checkpoint else config["FINAL_SAVE_DIR"]
    )
    model.load_weights(weights).expect_partial()

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

        np.savez_compressed(
            args.save_file, all_true=all_true, all_yhat=all_yhat, imageids=imageids
        )

    else:
        # the simpler approach, simply eval the test dataset
        model.evaluate(test_ds)


if __name__ == "__main__":
    main()
