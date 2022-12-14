import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# from collections import namedtuple, deque
from itertools import count

# from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

# import torch.nn.functional as F
# import torchvision.transforms as T

from inputextract import get_screen
from dqnmodule import DQN
from replaymemory import ReplayMemory, Transition
from utils import (
    check_network_identical,
    check_network_weights_loaded,
    estimate_training_time,
)


if gym.__version__ < "0.26":
    env = gym.make(
        "CartPole-v0", new_step_api=True, render_mode="single_rgb_array"
    ).unwrapped
else:
    env = gym.make("CartPole-v0", render_mode="rgb_array").unwrapped

env.reset()  # important to call before you do other stuff with env

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:", device)

# ========= Hyperparameters and utilities ================

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 100  # important, too small may cause unstable

WEIGHT_PATH = "weights.pt"
IS_TRAINING = False

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)

if os.path.exists(WEIGHT_PATH):
    print("[info] find weights file, policy_net load weights")
    policy_net.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    EPS_START = EPS_END

    assert check_network_weights_loaded(policy_net, WEIGHT_PATH)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

assert check_network_identical(policy_net, target_net)


optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

# will select an action accordingly to an epsilon greedy policy.
# Simply put, we???ll sometimes use our model for choosing the action,
#   and sometimes we???ll just sample one uniformly.
# The probability of choosing a random action will start at EPS_START and will
#   decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
def select_action(state):
    global steps_done, train_info
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    # train_info["eps_threshold"] = eps_threshold

    # print(eps_threshold, steps_done) # from EPS_START , decay to EPS_END
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


episode_durations = []
episode_loss = []

# a helper for plotting the durations of episodes,
#   along with an average over the last 100 episodes (the measure used in the official evaluations).
# The plot will be underneath the cell containing the main training loop, and will update after every episode.
def plot_durations():
    global train_info

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    loss_t = torch.tensor(episode_loss, dtype=torch.float)

    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    plt.plot(loss_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        train_info["latest mean"] = round(means[-1].item(), 2)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# ============= TRAINING LOOP ====================

# function that performs a single step of the optimization.
# It first samples a batch, concatenates all the tensors into a single one, computes Q(s_t, a_t) and V(s_{t+1}) = max_a Q(s_{t+1}, a),
# and combines them into our loss.
# By definition we set V(s) = 0 if s is a terminal state.
# We also use a target network to compute V(s_{t+1}), for added stability.
# The target network has its weights kept frozen most of the time, but
#   is updated with the policy network???s weights every so often.
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


num_episodes = 10000

train_info = {}

for i_episode in range(num_episodes):
    train_info["i_episode"] = i_episode
    train_info["estimate time"] = estimate_training_time(i_episode, num_episodes)

    # Initialize the environment and state
    env.reset()
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    state = current_screen - last_screen

    epoch_loss = 0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        epoch_loss += float(optimize_model())
        if done:
            episode_durations.append(t + 1)
            episode_loss.append(epoch_loss / (t + 1))

            train_info["latest loss"] = round(episode_loss[-1], 2)

            plot_durations()

            if i_episode > 0 and i_episode % 10 == 0:
                import json

                print(json.dumps(train_info, sort_keys=True), end="\r")
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        # save
        torch.save(policy_net.state_dict(), WEIGHT_PATH)
        # load
        target_net.load_state_dict(policy_net.state_dict())
        # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
        # Failing to do this will yield inconsistent inference results.
        target_net.eval()

        # assert check_network_identical(policy_net, target_net)
        # assert check_network_weights_loaded(policy_net, WEIGHT_PATH)


print("Complete")
env.render()
env.close()
plt.ioff()
plt.show()
