import os
import random
import numpy as np

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def get_rng(seed):
    # return numpy RandomState-like with randint etc.
    rng = np.random.RandomState(seed)
    return rng
