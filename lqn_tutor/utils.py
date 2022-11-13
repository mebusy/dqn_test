from hashlib import md5
from io import BytesIO
import torch
import time
import numpy as np


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t):
        self.epsilon = np.interp(t, [0, self.nsteps], [self.eps_begin, self.eps_end])


def np2torch(x, device, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x


def check_network_identical(network1, network2):
    """Check if two networks are identical.

    Args:
        network1 (torch.nn.Module): The first network.
        network2 (torch.nn.Module): The second network.

    Returns:
        bool: True if the two networks are identical, False otherwise.

    """
    buffer = BytesIO()
    torch.save(network1.state_dict(), buffer)
    md5_1 = md5(buffer.getbuffer()).hexdigest()

    buffer = BytesIO()
    torch.save(network2.state_dict(), buffer)
    md5_2 = md5(buffer.getbuffer()).hexdigest()

    return md5_1 == md5_2


def check_network_weights_loaded(network, weights_file):
    """Check if the network is identical to the weights file.

    Args:
        network (torch.nn.Module): The network.
        weights_file (str): The path to the weights file.

    Returns:
        bool: True if the network is identical to the weights file, False
            otherwise.

    """
    buffer = BytesIO()
    torch.save(network.state_dict(), buffer)
    md5_1 = md5(buffer.getbuffer()).hexdigest()

    with open(weights_file, "rb") as f:
        b = f.read()  # read file as bytes
        md5_2 = md5(b).hexdigest()
    return md5_1, md5_2


import time
import datetime

_last_tick = None
_avg_epoch_time = 0.00001
# invoke at the beginning of each epoch
def estimate_training_time(i_episode, total_episode):
    global _last_tick, _avg_epoch_time

    if not _last_tick:  # i_episode == 0
        _last_tick = time.time()
        return "estimating..."

    # i_episode >=1
    now = time.time()
    dt = now - _last_tick
    _last_tick = now

    _avg_epoch_time += (dt - _avg_epoch_time) / i_episode

    remaing_training_time = int((total_episode - i_episode) * _avg_epoch_time)
    return str(datetime.timedelta(seconds=remaing_training_time))
