import os
from itertools import count

import gym
import matplotlib.pyplot as plt

# import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replaymemory import ReplayMemory, Transition
from torchsummary import summary
from utils import (
    LinearSchedule,
    check_network_identical,
    check_network_weights_loaded,
    estimate_training_time,
    np2torch,
)

# --------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --------------------------------------------

if gym.__version__ < "0.26":
    env = gym.make(
        "CartPole-v0", new_step_api=True, render_mode="single_rgb_array"
    ).unwrapped
else:
    env = gym.make("CartPole-v0", render_mode="rgb_array").unwrapped

env.reset()  # important to call before you do other stuff with env


# ========= Hyperparameters and utilities ================

BATCH_SIZE = 512

GAMMA = 0.99
TARGET_UPDATE = 50  # important, too small may cause unstable

WEIGHT_PATH = "weights.pt"


# Get number of actions from gym action space
n_actions = env.action_space.n

hidden_layer_size = 512

for i in range(2):
    model = nn.Sequential(
        nn.Linear(4, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, n_actions),
    ).to(device)
    if i == 0:
        policy_net = model
    else:
        target_net = model
        summary(model, (4,), batch_size=BATCH_SIZE, device=device)


if os.path.exists(WEIGHT_PATH):
    print("[info] find weights file, policy_net load weights")
    policy_net.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
    assert check_network_weights_loaded(policy_net, WEIGHT_PATH)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
assert check_network_identical(policy_net, target_net)


optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


# will select an action accordingly to an epsilon greedy policy.
# Simply put, we’ll sometimes use our model for choosing the action,
#   and sometimes we’ll just sample one uniformly.
def select_action(state, t):
    eps_threshold = eps_schedule.epsilon

    if np.random.random() > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            # [[random.randrange(n_actions)]], device=device, dtype=torch.long
            [[env.action_space.sample()]],
            device=device,
            dtype=torch.long,
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


# ============= TRAINING LOOP ====================

# function that performs a single step of the optimization.
# It first samples a batch, concatenates all the tensors into a single one,
#   computes Q(s_t, a_t) and V(s_{t+1}) = max_a Q(s_{t+1}, a),
# and combines them into our loss.
# By definition we set V(s) = 0 if s is a terminal state.
# We also use a target network to compute V(s_{t+1}), for added stability.
# The target network has its weights kept frozen most of the time, but
#   is updated with the policy network’s weights every so often.
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

    # torch.gather(dim, index)  : Gathers values along an axis specified by dim.
    #     out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    #     out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    #     out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    # 就是将dim维度的索引改成index，其他不变
    # example:
    # >>> t = torch.tensor([[1, 2], [3, 4]])
    # >>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
    # tensor([[ 1,  1],   # t[0][1] = t[0][ index[0][1] ] = t[0][0] = 1
    #         [ 4,  3]])  # t[1][0] = t[1][ index[1][0] ] = t[1][1] = 4, t[1][1] = t[1][ index[1][1] ] = t[1][0] = 3
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
    # unsqueeze(dim),  e.g. (4, 2) -> unsqueeze(1) -> (4, 2, 1)
    # Returns a new tensor with a dimension of size one inserted at the specified position
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    # add by mebusy
    for group in optimizer.param_groups:
        group["lr"] = lr_schedule.epsilon

    optimizer.step()

    return loss


num_episodes = 800

train_info = {}


def getState(obs):
    # unsqueeze(dim),  e.g. (4, 2) -> unsqueeze(0) -> (1, 4, 2)
    # Returns a new tensor with a dimension of size one inserted at the specified position
    return np2torch(obs, device).unsqueeze(0)


lr_schedule = LinearSchedule(1e-2, 1e-4, num_episodes)
eps_schedule = LinearSchedule(0.5, 0.05, num_episodes)

for i_episode in range(num_episodes):
    train_info["i_episode"] = i_episode
    train_info["estimate time"] = estimate_training_time(i_episode, num_episodes)

    # Initialize the environment and state
    obs, info = env.reset()
    state = getState(obs)

    epoch_loss = 0
    for t in count():
        # Select and perform an action
        action = select_action(state, t)
        next_obs, reward, done, _, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        next_state = getState(next_obs)
        if done:
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

    eps_schedule.update(i_episode)
    lr_schedule.update(i_episode)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        # save
        # torch.save(policy_net.state_dict(), WEIGHT_PATH)
        # load
        target_net.load_state_dict(policy_net.state_dict())
        # Remember that you must call model.eval() to set dropout and batch normalization layers to
        #   evaluation mode before running inference.
        # Failing to do this will yield inconsistent inference results.
        target_net.eval()

        # assert check_network_identical(policy_net, target_net)
        # assert check_network_weights_loaded(policy_net, WEIGHT_PATH)


print("Complete")
env.render()
env.close()
plt.ioff()
# plt.show()

# =================================================================================================

from fcutils import dumpModelParams

dumpModelParams(model)
