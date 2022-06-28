import argparse
import yaml
import os
import torch
import torch.nn as nn

from utils.dataloader import get_dataloader_and_vocab
from utils.trainer import Trainer
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_config,
    save_vocab,
)


def train(config):

    os.makedirs(config["model_dir"])

    # Create the dataloaderSSS (get_dataloader_and_vocab)

    # Print some stats about the corpus (at least the vocab size)

    # Instantiate the model

    # Instantiate the loss function

    # Instantiate the optimiser, set it properly and maybe a LR scheduler ?

    # We prepare the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We create the Trainer (utils.Trainer)

    # We train
    print("Training finished.")

    # We save (trainer)
    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)
