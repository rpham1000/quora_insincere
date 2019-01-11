# https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
# standard imports
import random
import os
import numpy as np
import pandas as pd

# text preprocessing


# pytorch imports
import torch
import torch.nn as nn
import torch.utils.data


# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# progress bars
from tqdm import tqdm
tqdm.pandas()


class TrollHunter(object):
    def __init__(self,
                 train_x,
                 train_y,
                 test_x,
                 embedding_filename,
                 seed):
        """Insincere question detector for Quora

        Pytorch binary text classifier with

        Args:
            arg: [description]
        """
        self.training_data_x = train_x
        self.training_data_y = train_y
        self.seed = seed

        self.set_random_seed()

    def set_random_seed(self):
        """Make results deterministic

        https://www.kaggle.com/bminixhofer/deterministic-neural-networks-using-pytorch/notebook
        """
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def vectorize_words(self):
        pass

    def vectorize_sentences(self):
        pass

    def train_model(self):
        pass

    def test_model(self):
        pass


def main():
    input_filename = ''
    embedding_filename = ''
    output_filename = ''


if __name__ == '__main__':
    main()