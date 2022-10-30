import random
from collections import namedtuple, deque

# a named tuple representing a single transition in our environment.
# It essentially maps (state, action) pairs to their (next_state, reward) result,
#   with the state being the screen difference image as described later on.
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# a cyclic buffer of bounded size that holds the transitions observed recently.
# It also implements a .sample() method for selecting a random batch of transitions for training.
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    # auto filled by copilot
    memory = ReplayMemory(100)
    for i in range(100):
        memory.push(i, i, i, i)

    print(len(memory))
    print(memory.sample(10))

    for i in range(100):
        memory.push(i, i, i, i)

    print(len(memory))
    print(memory.sample(10))
