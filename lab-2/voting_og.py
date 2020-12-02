from numpy import loadtxt

import argparse
import _pickle as cp
import numpy as np


def load_raw_data(filename):
    raw_data = loadtxt(filename, delimiter=',', skiprows=0, dtype=str)
    return raw_data


def process_data(raw_data, include_missing=False):
    N_data = raw_data.shape[0]
    N_features = raw_data.shape[1] - 1

    X = np.zeros([N_data, N_features])
    y = np.zeros(N_data)

    complete = np.ones(N_data, dtype=bool)

    # Add data as np arrays
    for i in range(N_data):
        y[i] = 0 if raw_data[i][0] == "b'republican'" else 1
        for j in range(1, N_features + 1):
            if raw_data[i][j] == "b'y'":
                X[i, j - 1] = 1.
            elif raw_data[i][j] == "b'n'":
                X[i, j - 1] = 0.
            else:
                complete[i] = False
                X[i, j - 1] = 2.

    # Shuffle data before storing
    np.random.seed(0)
    shuffle = np.random.permutation(N_data)
    X = X[shuffle]
    y = y[shuffle]
    complete = complete[shuffle]

    if include_missing is True:
        return (X, y)
    else:
        return (X[complete], y[complete])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-missing', dest='include_missing',
                        action='store_true')
    parser.set_defaults(include_missing=False)
    args = parser.parse_args()

    if args.include_missing is True:
        filename = 'voting-full'
    else:
        filename = 'voting'

    input_filename = 'house-votes-84.data'

    raw_data = load_raw_data(input_filename)
    voting = process_data(raw_data, include_missing=args.include_missing)

    ff_file = open(filename + '.pickle', 'wb')
    cp.dump(voting, ff_file)
