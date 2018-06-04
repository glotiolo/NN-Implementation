import pickle
import gzip
import numpy as np

def load_mnist_data():
    """
    Return the MNIST data as (train_data, valid_data): train_data contains 
    50,000 tuples (x, y) and valid_data contains 10,000 tuples (x, y).  
    In each tuple, x is a 784 x 1 numpy array of floats between 0 and 1 
    representing  the pixels of the 28 x 28 input image of a hand-written 
    digit (0.0=white, 1.0=black).  y is the label (0..9).
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = pickle.load(f, encoding='latin1')
    f.close()
    train_data   = [(np.reshape(x, (784, 1)), y)
                       for x, y in zip(data[0][0], data[0][1])]
    valid_data = [(np.reshape(x, (784, 1)), y) 
                       for x, y in zip(data[1][0], data[1][1])]
    return (train_data, valid_data)
