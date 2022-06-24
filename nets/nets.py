import tensorflow as tf
from tensorflow import keras


def make_neural_network(base_arch_name, weights, image_size, dropout_pct, n_classes, input_dtype, train_full_network, ckpt):
    image_size_with_channels = image_size + [3]
    base_arch = keras.applications.Xception if base_arch_name == "xception" else None
    if not base_arch:
        print("Unsupported base architecture.")
        return None

    inputs = keras.layers.Input(shape=image_size_with_channels, dtype=input_dtype)
    base_model = base_arch(input_shape=image_size_with_channels, weights=None, include_top=False)

    if ckpt is not None:
        base_model.load_weights(ckpt)
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(n_classes, name="dense_logits")(x)
    output = keras.layers.Activation("softmax", dtype="float32", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=output)

    return model
