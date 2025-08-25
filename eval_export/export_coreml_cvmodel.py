import os

import click
import coremltools as ct
import tensorflow as tf


@click.command()
@click.option("--keras_model", type=str, required=True)
@click.option("--quantization_nbits", type=int, required=False)
@click.option("--output", type=str, required=True)
def main(keras_model, quantization_nbits, output):
    keras_model = tf.keras.models.load_model(keras_model) 
    image_input = ct.ImageType(
        shape=(1, 299, 299, 3,),
        scale=1.0/255.0,
    )
    coreml_model = ct.convert(
        keras_model,
        inputs=[image_input],
        minimum_deployment_target=ct.target.iOS13
    )

    if quantization_nbits is None:
        coreml_model.save(output)
    else:
        quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
            coreml_model, quantization_nbits
        )
        quantized_model.save(output)


if __name__ == "__main__":
    main()


