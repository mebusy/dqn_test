# import numpy as np
from npFC import loadTorchWeights, TwoLayerNet

from dataload import loader_val


def check_accuracy_part34(loader, net):
    if loader.dataset.train:
        print("Checking accuracy on validation set")
    else:
        print("Checking accuracy on test set")
    num_correct = 0
    num_samples = 0

    for x, y in loader:
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        scores = net.forward(x)
        # print(scores.shape)
        preds = scores.argmax(axis=1)
        num_correct += (preds == y).sum()
        num_samples += preds.size
    acc = float(num_correct) / num_samples
    print("Got %d / %d correct (%.2f)" % (num_correct, num_samples, 100 * acc))


# -------------------------- main -------------------------------------------

if __name__ == "__main__":

    params = loadTorchWeights("./params")

    net = TwoLayerNet(params=params)
    # scores = net.forward(np.random.randn(2, 3, 32, 32))
    # print(scores)

    check_accuracy_part34(loader_val, net)
