import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

from ignite.engine import Events, Engine
# from ignite.metrics import CategoricalAccuracy, Precision, Recall
# from metrics import Loss
#
# from torchtext import data
#
# from handlers import ModelLoader, ModelCheckpoint
# from preprocessing import cleanup_text
# from helper import create_supervised_evaluator
# from pydoc import locate

from utils import load_yaml
from models import RNNClassifier, StackedCRNNClassifier

PARSER = argparse.ArgumentParser(description="Twitter Sentiment Analysis")
PARSER.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
PARSER.add_argument(
    "--dataset",
    type=str,
    default="../data/training.1600000.processed.noemoticon.csv",
    help="""Path for your dataset.
    As this package uses torch text to load the data, please
    follow the format by providing the path and filename without its
    extension""",
)
PARSER.add_argument("--batch_size", type=int, default=16, help="The number of batch size for every step")
PARSER.add_argument("--log_interval", type=int, default=100)
PARSER.add_argument("--save_interval", type=int, default=500)
PARSER.add_argument("--validation_interval", type=int, default=500)
PARSER.add_argument(
    "--char_level",
    help="Whether to use the model with "
    "character level or word level embedding. Specify the option "
    "if you want to use character level embedding",
)
PARSER.add_argument(
    "--model_config",
    type=str,
    default="config/rnn.yml",
    help="Location of model config",
)
PARSER.add_argument("--model_dir", type=str, default="models", help="Location to save the model")

ARGS = PARSER.parse_args()

if __name__ == "__main__":
    #Loading config files
    model_config = load_yaml(ARGS.model_config)
    device = -1  # Use CPU as default

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = None
    print(model_config)