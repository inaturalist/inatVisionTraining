import math
import os

import click
import numpy as np
import tensorflow as tf

@click.command()
@click.option("--keras_model", type=str, required=True)
@click.option("--output", type=str, required=True)
def main(keras_model, output):
    keras_model = tf.keras.models.load_model(keras_model)
    
    # scale inputs so mobile can feed in range 0,255
    # model expects range in 0,1
    inputs = tf.keras.layers.Input(shape=keras_model.input_shape[1:])
    x = tf.keras.layers.Rescaling(scale=1.0/255.0)(inputs)
    outputs = keras_model(x)
    keras_model_with_scaling = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model_with_scaling)
    # default options include quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    print("Saving tflite model")
    export_file = "model_exports/INatVision_2_23_mobile_8bit.tflite"
    with open(output, 'wb') as f:
        f.write(quantized_model)

   

    
if __name__ == "__main__":
    main()


