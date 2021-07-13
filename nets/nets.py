import tensorflow as tf
from tensorflow import keras


def make_neural_network(base_arch_name, image_size, dropout_pct, n_classes, input_dtype, train_full_network):
    image_size_with_channels = image_size + [3]
    base_arch = keras.applications.Xception if base_arch_name == "xception" else None
    if not base_arch:
        print("Unsupported base architecture.")
        return None

    input_layer = keras.layers.Input(shape=image_size_with_channels, dtype=input_dtype)
    base_model = base_arch(input_tensor=input_layer, weights="imagenet", include_top=False)
    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
    dropout = keras.layers.Dropout(dropout_pct)(avg)
    x = keras.layers.Dense(n_classes, name="dense_logits")(dropout)
    output = keras.layers.Activation("softmax", dtype="float32", name="predictions")(x)
    model = keras.Model(inputs=base_model.input, outputs=output)

    if train_full_network:
        for layer in base_model.layers:
            layer.trainable = True

    return model
