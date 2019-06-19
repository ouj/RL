import tensorflow as tf
import numpy as np

def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

