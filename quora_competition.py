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
# imports for preprocessing the questions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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

    def vectorize_words(self,
                        embed_size,
                        max_features,
                        max_len):
        """[summary]

        https://www.kaggle.com/gmhost/gru-capsule
        """

        puncts = [
            ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$',
            '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~',
            '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°',
            '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à',
            '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░',
            '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹',
            '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄',
            '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘',
            '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³',
            '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

        def clean_text(x):
            x = str(x)
            for punct in puncts:
                # TODO: What is this little script 'f'?
                x = x.replace(punct, f' {punct} ')
            return x

        # Lowercase all
        self.train_df["question_text"] = self.train_df[
            "question_text"].str.lower()
        self.test_df["question_text"] = self.test_df[
            "question_text"].str.lower()

        # Remove punctuation in punctuation list
        self.train_df["question_text"] = self.train_df[
            "question_text"].apply(lambda x: clean_text(x))
        self.test_df["question_text"] = self.test_df[
            "question_text"].apply(lambda x: clean_text(x))

        # Fill up the missing values
        # TODO: Why are there na values?
        x_train = self.train_df["question_text"].fillna("_##_").values
        x_test = self.test_df["question_text"].fillna("_##_").values

        # Tokenize the sentences
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(x_train))
        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)

        # Pad the sentences
        x_train = pad_sequences(x_train, maxlen=max_len)
        x_test = pad_sequences(x_test, maxlen=max_len)

        # Get the target values
        y_train = self.train_df['target'].values

        return x_train, y_train, x_test, tokenizer

    def vectorize_sentences(self):
        pass

    def train_model(self):
        pass

    def predict(self):
        pass

    def test_model(self):
        pass


class Embedding(object):
    def __init__(self,
                 embedding_filename):
        """[summary]

        [description]

        Args:
            embedding_filename: [description]
        """
        self.embedding_filename = embedding_filename


class Glove(Embedding):
    def __init__(self,
                 embedding_filename,
                 word_index,
                 max_features):
        """[summary]

        What is going on here?!?!

        Args:
            embedding_filename: [description]
        """
        super().__init__(embedding_filename=embedding_filename)

        # TODO: What is this little asterick?
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')[:300]
        embeddings_index = dict(
            get_coefs(*o.split(" ")) for o in open(embedding_filename))

        all_embs = np.stack(embeddings_index.values())
        emb_mean = -0.005838499
        emb_std = 0.48782197
        embed_size = all_embs.shape[1]

        nb_words = min(max_features, len(word_index))
        embedding_matrix = np.random.normal(
            emb_mean,
            emb_std,
            (nb_words, embed_size))
        for word, i in word_index.items():
            if i >= max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self.matrix = self.embedding_matrix


class FastText(Embedding):
    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg


class Para(Embedding):
    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg


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

    # Text preprocessing
    embed_size = 300  # how big is each word vector
    # how many unique words to use (i.e num rows in embedding vector)
    max_features = 95000
    max_len = 70  # max number of words in a question to use
    x_train, y_train, x_test, tokenizer = han.vectorize_words(
        embed_size=embed_size,
        max_features=max_features,
        max_len=max_len)

    hand_in = Glove(embedding_filename)


if __name__ == '__main__':
    main()


# TODO: Ideas to test
# 1) Genetic algorithm sentence generator, selecting for insincere with NN
# 2) Vecotorizing sentence in manner that preservers word order
#       * Switching to CNN
