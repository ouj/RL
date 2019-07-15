import tensorflow as tf
import numpy as np

def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)

def atleast_4d(x):
    return np.expand_dims(np.atleast_3d(x), axis=0) if x.ndim < 4 else x

def get_saver(session, checkpoint_dir):
    saver = tf.train.Saver()
    last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if last_checkpoint is not None:
        saver.restore(session, last_checkpoint)
        print("Restored last checkpoint", last_checkpoint)
    return saver
