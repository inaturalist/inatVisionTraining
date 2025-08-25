#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
from datetime import datetime

import click
import numpy as np
import os
import pandas as pd
import requests
import tensorflow as tf
from tqdm.auto import tqdm


def make_model_load_checkpoint(num_classes, checkpoint):
    image_size_with_channels = [299, 299] + [3]
    base_arch = tf.keras.applications.Xception
    inputs = tf.keras.layers.Input(
        shape=image_size_with_channels, 
        dtype=tf.float32
    )
    base_model = base_arch(
        input_shape=image_size_with_channels, 
        weights=None, 
        include_top=False
    )
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="final_global_pool")(x)
    x = tf.keras.layers.Dense(num_classes, name="dense_logits")(x)
    output = tf.keras.layers.Activation("softmax", dtype="float32", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.load_weights(checkpoint).expect_partial()
    return model


def make_fact_model_load_checkpoint(num_classes, fact_rank, checkpoint):
    image_size_with_channels = [299, 299] + [3]
    base_arch = tf.keras.applications.Xception
    inputs = tf.keras.layers.Input(
        shape=image_size_with_channels, 
        dtype=tf.float32
    )
    base_model = base_arch(
        input_shape=image_size_with_channels, 
        weights=None, 
        include_top=False
    )
    x = base_model(inputs, training=False)
    
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    svd_u = tf.keras.layers.Conv2D(fact_rank, [1, 1])(x)
    logits = tf.keras.layers.Conv2D(num_classes, [1, 1])(svd_u)
    logits = tf.keras.layers.Reshape([num_classes])(logits)
  
    output = tf.keras.layers.Activation("softmax", dtype="float32", name="predictions")(logits)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.load_weights(checkpoint)
    return model


@click.command()
@click.option("--checkpoint_path", type=str, required=True)
@click.option("--num_classes", type=int, required=True)
@click.option("--fact_rank", type=int, required=False)
@click.option("--output_path", type=str, required=True)
def main(checkpoint_path, num_classes, fact_rank, output_path):
    if fact_rank is not None:
        model = make_fact_model_load_checkpoint(
            num_classes,
            fact_rank,
            checkpoint_path
        )
    else:
        model = make_model_load_checkpoint(
            num_classes,
            checkpoint_path
        )

    model.save(output_path)



if __name__ == "__main__":
    main()

