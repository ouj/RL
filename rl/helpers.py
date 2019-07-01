import tensorflow as tf
import numpy as np

def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

def atleast_4d(x):
    return np.expand_dims(np.atleast_3d(x), axis=0) if x.ndim < 4 else x