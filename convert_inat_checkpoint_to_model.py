#!/usr/bin/env python
# coding: utf-8

import click
import pandas as pd
import tensorflow as tf

def make_model_load_checkpoint(num_classes, checkpoint, fact_rank=None):
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
    if fact_rank is None:
        x = tf.keras.layers.GlobalAveragePooling2D(name="final_global_pool")(x)
        x = tf.keras.layers.Dense(num_classes, name="dense_logits")(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="final_global_pool")(x)
        x = tf.keras.layers.Conv2D(fact_rank, [1, 1], name="svd_u")(x)
        x = tf.keras.layers.Conv2D(num_classes, [1, 1], name="logits")(x)
        x = tf.keras.layers.Reshape([num_classes])(x)
 
    output = tf.keras.layers.Activation("softmax", dtype="float32", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.load_weights(checkpoint).expect_partial()
    return model


@click.command()
@click.option("--taxonomy_path", type=str, required=True)
@click.option("--checkpoint_path", type=str, required=True)
@click.option("--factorize_rank", type=int, required=False)
@click.option("--output_path", type=str, required=True)
def main(taxonomy_path, checkpoint_path, factorize_rank, output_path):
    tax = pd.read_csv(taxonomy_path)
    num_classes = len(tax[~tax.leaf_class_id.isna()])

    model = make_model_load_checkpoint(
        num_classes,
        checkpoint_path,
        factorize_rank
    )

    model.save(output_path)

if __name__ == "__main__":
    main()    


