import pandas as pd
import tensorflow as tf
from functools import partial

AUTOTUNE = tf.data.experimental.AUTOTUNE


# augments
# currently not using rotate
def _rotate(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    rotate_amt = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, rotate_amt)
    return x, y


def _flip(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    # right left only
    x = tf.image.random_flip_left_right(x)
    return x, y


def _color(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x, y


def _random_crop(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(x),
        bounding_boxes=bbox,
        area_range=(0.08, 1.0),
        aspect_ratio_range=(0.75, 1.33),
        max_attempts=100,
        min_object_covered=0.1,
    )
    x = tf.slice(x, begin, size)

    return x, y


def _decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # no resizing, we may augment which will crop.
    # we resize _after_ the augments pass
    return img


def _process(file_path, label, num_classes):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = _decode_img(img)
    # 1 hot encode the label for dense
    label = tf.one_hot(label, num_classes)
    return img, label


def _load_dataframe(dataset_csv_path):
    df = pd.read_csv(dataset_csv_path)

    # sort the dataset
    df = df.sample(frac=1, random_state=42)
    return df


def _prepare_dataset(
    ds,
    image_size=(299, 299),
    batch_size=32,
    repeat_forever=True,
    augment=False,
):
    # do transforms for augment or not
    if augment:
        # crop 100% of the time
        ds = ds.map(lambda x, y: _random_crop(x, y), num_parallel_calls=AUTOTUNE)
        # flip 50% of the time
        # the function already flips 50% of the time, so we call it 100% of the time
        ds = ds.map(lambda x, y: _flip(x, y), num_parallel_calls=AUTOTUNE)
        # do color 30% of the time
        ds = ds.map(
            lambda x, y: tf.cond(
                tf.random.uniform([], 0, 1) > 0.7, lambda: _color(x, y), lambda: (x, y)
            ),
            num_parallel_calls=AUTOTUNE,
        )
        # resize to image size expected by network
        ds = ds.map(lambda x, y: (tf.image.resize(x, image_size), y))
        # make sure the color transforms haven't move any of the pixels outside of [0,1]
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(x, 0, 1), y), num_parallel_calls=AUTOTUNE
        )
    else:
        # central crop
        ds = ds.map(lambda x, y: (tf.image.central_crop(x, 0.875), y))
        # resize to image size expected by network
        ds = ds.map(lambda x, y: (tf.image.resize(x, image_size), y))

    # Repeat forever
    if repeat_forever:
        ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def make_dataset(
    path,
    label_column_name,
    image_size=(299, 299),
    batch_size=32,
    shuffle_buffer_size=10_000,
    repeat_forever=True,
    augment=False,
):
    df = _load_dataframe(path)
    num_examples = len(df)
    num_classes = len(df[label_column_name].unique())

    ds = tf.data.Dataset.from_tensor_slices((df["filename"], df[label_column_name]))

    ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    process_partial = partial(_process, num_classes=num_classes)
    ds = ds.map(process_partial, num_parallel_calls=AUTOTUNE)

    ds = _prepare_dataset(
        ds,
        image_size=image_size,
        batch_size=batch_size,
        repeat_forever=repeat_forever,
        augment=augment,
    )

    return (ds, num_examples)
