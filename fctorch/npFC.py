#!python3
import numpy as np

# -------------------------- forward/backward-------------------------------------------


def affine_forward(x, w, b):
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


# -------------------------- FC Net -------------------------------------------


class TwoLayerNet(object):
    """
    Input dimension: D, hidden dimension  H,  C classes.
    Architecure:  affine - relu - affine - softmax.
    """

    def __init__(self, params=None):
        """
        Initialize a new network.

        Inputs:
        - param: An integer giving the number of classes to classify
        """
        if params is None:
            raise ValueError("Params is None")

        self.params = params

        pass

    def loss(self, X):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        _scores1, cache1 = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        scores, cache2 = affine_forward(_scores1, self.params["W2"], self.params["b2"])

        # If y is None then we are in test mode so just return scores
        return scores


# -------------------------- main -------------------------------------------

if __name__ == "__main__":
    import os

    # load trained pytorch weights
    weight_files = sorted([f for f in os.listdir("./params") if f.endswith(".np")])
    params = {}
    for layer in range(2):
        for f in weight_files[layer * 2 : layer * 2 + 2]:
            # print(f)
            if f.find(".bias") != -1:
                params[f"b{layer+1}"] = np.load("./params/" + f)
            elif f.find(".weight") != -1:
                # transpose to match our FCnet
                params[f"W{layer+1}"] = np.load("./params/" + f).T
            else:
                raise ValueError("missing weight file")

    # print(params.items())
    # print("params loaded:", {k: v.shape for k, v in params.items()})

    # note, those commented code are not transposed
    # H, D = params["W1"].shape
    # C = params["b2"].shape[0]
    # net = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C)

    # test
    net = TwoLayerNet(params=params)
    scores = net.loss(np.random.randn(2, 3, 32, 32))
    print(scores)
    pass
