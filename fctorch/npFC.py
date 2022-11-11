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

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        # std = weight_scale.  PS. NOT variance
        self.params["W1"] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params["W2"] = np.random.randn(hidden_dim, num_classes) * weight_scale

        self.params["b1"] = np.zeros(hidden_dim)
        self.params["b2"] = np.zeros(num_classes)

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

    weight_files = sorted([f for f in os.listdir("./params") if f.endswith(".np")])
    params = {}
    for layer in range(2):
        for f in weight_files[layer * 2 : layer * 2 + 2]:
            # print(f)
            if f.find(".bias") != -1:
                params[f"b{layer+1}"] = np.load("./params/" + f)
            elif f.find(".weight") != -1:
                params[f"w{layer+1}"] = np.load("./params/" + f)
            else:
                raise ValueError("missing weight file")

    # print(params.items())
    print("params loaded:", {k: v.shape for k, v in params.items()})

    net = TwoLayerNet(input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10)
    scores = net.loss(np.random.randn(2, 3, 32, 32))
    print(scores)
    pass
