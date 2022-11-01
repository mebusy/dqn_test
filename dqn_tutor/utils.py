from hashlib import md5
from io import BytesIO
import torch


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

def check_network_weights_loaded( network, weights_file ):
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


    with open(weights_file,"rb") as f:
        b = f.read() # read file as bytes
        md5_2 = md5(b).hexdigest();
    return md5_1, md5_2



