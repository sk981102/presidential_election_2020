import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import torch
import yaml
import six
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

from collections import namedtuple

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"


def load_yaml(config_path):
    if not isinstance(config_path, six.string_types):
        raise ValueError("Got {}, expected string", type(config_path))
    else:
        with open(config_path, "r") as yaml_file:
            config = yaml.load(yaml_file)
            return config


def preprocess(text, stem=False):
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)

    return tokens


def decode_sentiment(label):
    return decode_map[int(label)]
