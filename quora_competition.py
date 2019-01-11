# http://www.deborahmesquita.com/2017-11-05/how-pytorch-gives-the-big-picture-with-deep-learning
# standard imports
import random
import os
import numpy as np
import pandas as pd

# text preprocessing
# http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
# https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings/comments
# https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings

# pytorch imports
import torch
import torch.nn as nn
import torch.utils.data


# cross validation and metrics
# TODO: Test stratified vs normal k folds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# progress bars
from tqdm import tqdm
tqdm.pandas()


class TrollHunter(object):
    def __init__(self,
                 train_filename,
                 test_filename,
                 embedding_filename,
                 output_filename,
                 seed=1234):
        """Insincere question detector for Quora

        Pytorch binary text classifier with

        Args:
            arg: [description]
        """
        self.train_df = pd.read_csv(train_filename)
        self.test_df = pd.read_csv(test_filename)
        print('Train data dimension: ', self.train_df.shape)
        print(self.train_df.head())
        print('Test data dimension: ', self.test_df.shape)
        print(self.test_df.head())

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
        """[summary]

        [description]
        """


    def vectorize_sentences(self):
        pass

    def train_model(self):
        pass

    def test_model(self):
        pass


def main():
    """[summary]

    [description]
    """
    # Input filenames
    train_filename = os.path.join(
        'input',
        'train.csv')
    test_filename = os.path.join(
        'input',
        'test.csv')
    embedding_filename = os.path.join(
        'embeddings',
        'glove.840B.300d',
        'glove.840B.300d.txt')
    output_filename = os.path.join(
        'output',
        'submission.csv')

    han = TrollHunter(
        train_filename=train_filename,
        test_filename=test_filename,
        embedding_filename=embedding_filename,
        output_filename=output_filename)

    han.vectorize_words()


if __name__ == '__main__':
    main()


# TODO: Ideas to test
# 1) Genetic algorithm sentence generator, selecting for insincere with NN
# 2) Vecotorizing sentence in manner that preservers word order
#       * Switching to CNN
