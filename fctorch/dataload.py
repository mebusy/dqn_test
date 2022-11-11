from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose(
    [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)


cifar10_train = dset.CIFAR10(
    "./datasets", train=True, download=True, transform=transform
)
loader_train = DataLoader(
    cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN))
)

# **acutally same as cifar10_train**
cifar10_val = dset.CIFAR10("./datasets", train=True, download=True, transform=transform)
# **but be different on subset**
loader_val = DataLoader(
    cifar10_val,
    batch_size=64,
    sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)),
)


if __name__ == "__main__":
    # print("cifar10_train: ", cifar10_train)
    # print("loader_train: ", loader_train)
    # print("cifar10_val: ", cifar10_val)
    print("loader_val: ", loader_val)

    for x, y in loader_val:
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        print("x: ", x)
        print("y: ", y)
