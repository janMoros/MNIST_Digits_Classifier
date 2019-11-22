import pickle
import glob
import os

def pkl_export(predictions, filename):
    """ Exports a given array into binary pkl format

    Args:
        predictions: array to save
        filename: string with file path: for instance preds.pkl
    """
    with open(filename, 'wb') as outfile:
        pickle.dump(predictions, outfile, protocol=2)


def pkl_import(filename):
    """ Imports a binary pickled file

    Args:
        filename: path to the file to import

    Returns: the file content.

    """
    with open(filename, 'rb') as infile:
        return pickle.load(infile, encoding="bytes")


def pkl_concat(path, out):
    """ Concatenates all pickle files in a given directory into another pickle file

    Args:
        path: the path where all the pickle files are stored
        out: the path where the resulting pickle file will be stored 
    """

    pkl_files = glob.glob(os.path.join(path, '*.pkl'))
    pkl_contents = []
    for pkl in pkl_files:
        pkl_contents.append(pkl_import(pkl))
    pkl_export(pkl_contents, out)


# Test
if __name__ == "__main__":

    import numpy as np

    a = np.arange(10, dtype=float)
    b = np.arange(10, dtype=float)
    pkl_export(predictions=a, filename="tmp.pkl")
    pkl_export(predictions=a, filename="tmp1.pkl")
    assert ((np.array(pkl_import("tmp.pkl")) == a[:]).all()) 
