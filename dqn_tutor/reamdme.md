
# key idea

1. present the state as the difference between the current screen patch and the previous one
2. Replay Memory
    - It stores the transitions that the agent observes
    - By sampling from it randomly, the transitions that build up a batch are decorrelated. 



- Transition: 
    - a named tuple representing a single transition in our environment.
    - It essentially maps (state, action) pairs to their (next_state, reward) result
    - `('state', 'action', 'next_state', 'reward')`
- ReplayMemory;
    - a cyclic buffer of bounded size that holds the transitions observed recently
    - It also implements a `.sample()` method for selecting a random batch of transitions for training.
