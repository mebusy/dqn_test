import os
import numpy as np


def dumpModelParams(model):
    out_params = "params"
    if not os.path.exists(out_params):
        os.makedirs(out_params, exist_ok=True)
    for name, param in model.named_parameters():
        # save param to file
        path = os.path.join(out_params, name + ".np")
        # torch.save(param, path)
        with open(path, "wb") as fp:
            print("saveing:", name, param.shape)
            np_data = param.data.cpu().numpy()
            np.save(fp, np_data)
            print("    to numpy:", np_data.shape)
