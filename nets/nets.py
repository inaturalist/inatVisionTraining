import tensorflow as tf
from tensorflow import keras


def make_neural_network(
    base_arch_name,
    weights,
    image_size,
    n_classes,
    input_dtype,
    train_full_network,
    ckpt,
    factorize=False,
    fact_rank=None,
    activation=None
):
    image_size_with_channels = image_size + [3]
    base_arch = keras.applications.Xception if base_arch_name == "xception" else None
    if not base_arch:
        print("Unsupported base architecture.")
        return None

    inputs = keras.layers.Input(shape=image_size_with_channels, dtype=input_dtype)

    base_model = base_arch(
        input_shape=image_size_with_channels, weights=weights, include_top=False
    )
    if train_full_network:
        # still don't want to train batchnorm layers
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                print(f"freezing layer {layer}.")
                layer.trainable = False
            else:
                print(f"unfreezing layer {layer}.") 
                layer.trainable = True
    else:
        base_model.trainable = False
    
    if ckpt is not None:
        base_model.load_weights(ckpt)

    x = base_model(inputs)
    
    if factorize and fact_rank is not None:
        x = keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
        svd_u = keras.layers.Conv2D(fact_rank, [1, 1])(x)
        logits = keras.layers.Conv2D(n_classes, [1, 1])(svd_u)
        logits = keras.layers.Reshape([n_classes])(logits)
    else:
        x = keras.layers.GlobalAveragePooling2D(name="final_global_pool")(x)
        logits = keras.layers.Dense(n_classes, name="dense_logits")(x)

    if activation is not None:
        output = keras.layers.Activation(
            activation,
            dtype="float32",
            name="predictions"
        )
        model = keras.Model(inputs=inputs, outputs=output)
    else:
        model = keras.Model(inputs=inputs, outputs=logits)
        
    return model
