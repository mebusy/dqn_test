# from builtins import range
# from builtins import object
import numpy as np

from .layers import *
from .layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
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

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # std = weight_scale.  PS. NOT variance
        self.params["W1"] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params["W2"] = np.random.randn(hidden_dim, num_classes) * weight_scale

        self.params["b1"] = np.zeros(hidden_dim)
        self.params["b2"] = np.zeros(num_classes)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
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
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        _scores1, cache1 = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        scores, cache2 = affine_forward(_scores1, self.params["W2"], self.params["b2"])
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        loss, smx_grad = softmax_loss(scores, y)  # loss of score
        # + loss of regularization
        # **loss won't be actually used in calculating gradients**
        loss += 0.5 * self.reg * ((W1 * W1).sum() + (W2 * W2).sum())

        dx2, dw2, db2 = affine_backward(smx_grad, cache2)
        grads["W2"] = dw2 + self.reg * W2
        grads["b2"] = db2

        dx1, dw1, db1 = affine_relu_backward(dx2, cache1)
        grads["W1"] = dw1 + self.reg * W1
        grads["b1"] = db1

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        all_input_output = (
            [input_dim] + hidden_dims + [num_classes]
        )  # len = num_layer + 1
        for i in range(self.num_layers):
            idx_layer = i + 1
            size_input = all_input_output[i]
            size_output = all_input_output[i + 1]
            # print( i, idx_layer, self.num_layers, all_input_output )
            self.params["W{}".format(idx_layer)] = (
                np.random.randn(size_input, size_output) * weight_scale
            )
            self.params["b{}".format(idx_layer)] = np.zeros(size_output)

            # initialization for batch norm
            if self.normalization is not None and i < self.num_layers - 1:
                self.params["gamma{}".format(idx_layer)] = np.ones(size_output)
                self.params["beta{}".format(idx_layer)] = np.zeros(size_output)

        pass

        # print( self.params.keys() )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layer_caches = {}

        # A LAYER means `{affine - [batch/layer norm] - relu - [dropout]}`
        for i in range(self.num_layers):
            idx_layer = i + 1  # key index of `W` and `b`

            # which forward function to use
            f = affine_forward if i == self.num_layers - 1 else affine_relu_forward
            if f == affine_relu_forward:
                if self.normalization == "batchnorm":  # batch norm
                    f = affine_bn_relu_forward
                elif self.normalization == "layernorm":  # batch norm
                    f = affine_ln_relu_forward

            W, b = (
                self.params["W{}".format(idx_layer)],
                self.params["b{}".format(idx_layer)],
            )
            # execute forward function, get score and cache
            # scores initialized as None if the 1st layer, and pass along to the next layer
            # print( len(self.bn_params), i )
            if (
                f == affine_bn_relu_forward or f == affine_ln_relu_forward
            ):  # batch/layer norm
                scores, cache = f(
                    scores if scores is not None else X,
                    W,
                    b,
                    self.params["gamma{}".format(idx_layer)],
                    self.params["beta{}".format(idx_layer)],
                    self.bn_params[i],
                )

            else:
                # layers without batch/layer norm
                scores, cache = f(scores if scores is not None else X, W, b)

            if (
                self.use_dropout and f != affine_forward
            ):  # user dropout , AND, not last affine layer
                scores, cache_dropout = dropout_forward(scores, self.dropout_param)
                # insert dropout cache
                cache += (cache_dropout,)

            # save cache for each layer
            layer_caches[idx_layer] = cache

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.
        #   ( dgamma, dbeta only ?  of for all gradients includes dx,dw ? )        #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, smx_grad = softmax_loss(scores, y)  # loss of score
        # + loss of regularization
        # **loss won't be actually used in calculating gradients**
        list_w_square = [
            (self.params["W{}".format(i + 1)] * self.params["W{}".format(i + 1)]).sum()
            for i in range(self.num_layers)
        ]
        loss += 0.5 * self.reg * (sum(list_w_square))

        upstream_grad = smx_grad
        for idx_layer in range(self.num_layers, 0, -1):
            cache = layer_caches[idx_layer]

            # print(idx_layer , len(cache), idx_layer == self.num_layers )
            W = self.params["W{}".format(idx_layer)]
            f = (
                affine_backward
                if idx_layer == self.num_layers
                else affine_relu_backward
            )

            if f == affine_relu_backward:  #  norm
                if self.normalization == "batchnorm":
                    f = affine_bn_relu_backward
                elif self.normalization == "layernorm":
                    f = affine_ln_relu_backward

            # dropout layer
            if (
                self.use_dropout and f != affine_backward
            ):  # user dropout , AND, not last affine layer
                cache, cache_dropout = cache[:-1], cache[-1]
                # print ( cache_dropout[0], cache_dropout[1] )
                dx = dropout_backward(upstream_grad, cache_dropout)
                # calc dx only, no parameter updating
                upstream_grad = dx

            if (
                f == affine_bn_relu_backward or f == affine_ln_relu_backward
            ):  # batch/layer norm

                dx, dw, db, dgamma, dbeta = f(upstream_grad, cache)

                grads["gamma{}".format(idx_layer)] = dgamma
                grads["beta{}".format(idx_layer)] = dbeta
            else:
                # layers without batch/layer norm
                dx, dw, db = f(upstream_grad, cache)

            grads["W{}".format(idx_layer)] = dw + self.reg * W
            grads["b{}".format(idx_layer)] = db

            upstream_grad = dx
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


# for batch norm
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)

    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)

    dx_bn, dgamma, dbeta = batchnorm_backward_alt(da, bn_cache)

    dx, dw, db = affine_backward(dx_bn, fc_cache)

    return dx, dw, db, dgamma, dbeta


# for layer norm
def affine_ln_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = layernorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)

    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_ln_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)

    dx_bn, dgamma, dbeta = layernorm_backward(da, bn_cache)

    dx, dw, db = affine_backward(dx_bn, fc_cache)

    return dx, dw, db, dgamma, dbeta
