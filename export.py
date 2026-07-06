import argparse
import os

import numpy as np
import tensorflow as tf
import yaml

from nets import nets


class ModelExporter:
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = self.config["CHECKPOINT_DIR"]
        self.output_file = os.path.join(
            config["FINAL_SAVE_DIR"], "INatVision.h5"
        )

    def load_model(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)

        model = nets.make_neural_network(
            base_arch_name="xception",
            image_size=self.config["IMAGE_SIZE"],
            n_classes=self.config["NUM_CLASSES"],
            factorize=self.config["FACTORIZE_FINAL_LAYER"] if "FACTORIZE_FINAL_LAYER" in self.config else False,
            fact_rank=self.config["FACT_RANK"] if "FACT_RANK" in self.config else None,
            weights=None,
            input_dtype=np.float32,
            ckpt=None,
            train_full_network=False
        )

        # Load the weights
        model.load_weights(latest).expect_partial()

        return model

    def export_to_hdf5(self):
        model = self.load_model()
        if model is not None:
            model.save(self.output_file, save_format='h5')
            print("Model successfully saved to {}".format(
                self.output_file
            ))
        else:
            print("Failed to load model.")


def main():
    # get command line args
    parser = argparse.ArgumentParser(description="Train an iNat model.")
    parser.add_argument(
        "--config_file", required=True, help="YAML config file for training."
    )
    args = parser.parse_args()

    # read in config file
    if not os.path.exists(args.config_file):
        print("No config file.")
        return
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # Example usage
    exporter = ModelExporter(config)
    exporter.export_to_hdf5()


if __name__ == "__main__":
    main()
