#!python3

"""
nn test
"""
from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt

from data_utils import get_CIFAR10_data


def rel_error(x, y):
    """returns relative error"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


data = get_CIFAR10_data()
for k, v in list(data.items()):
    print(("%s: " % k, v.shape))
