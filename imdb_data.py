"""
Creates a Data Frame of the imdb data. This is a reference data set
that should performe quit well. With this I check the implementation of my models
"""

import os
import glob
import numpy as np
from sklearn.utils import shuffle
import pandas as pd


def read_imdb_data(data_dir=r'C:\Users\hauer\Documents\Repositories\ml-templates\datasets\aclImdb'):
    data = {}
    labels = {}

    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []

            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            #CUSTOM: do not load all files
            counter = 0
            for f in files:
                counter += 1

                if counter > 2500:
                    break

                with open(f, encoding='utf8') as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)

            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                "{}/{} data size does not match labels size".format(data_type, sentiment)

    return data, labels




def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""

    # Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']

    # Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    

    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test


def get_imdb_data():
    data, labels = read_imdb_data()
    
    
    train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
    
    X = np.concatenate((train_X, test_X))
    y = np.concatenate((train_y, test_y))
    
    return X, y


if __name__ == "__main__":
    X, y = get_imdb_data()
    
    df = pd.DataFrame(data = {'title': X, 'Target': y})
    
    df.to_pickle(r'temp\imdb.pkl')

