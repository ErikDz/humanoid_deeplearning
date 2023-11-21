import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def compute_jump_reward(observation, info):
    z_coord_of_torso = observation[0]  # Assuming the z-coordinate of the torso indicates height
    velocity_z_coord_of_torso = observation[24]  # Assuming this is the vertical velocity of the torso

    # A simple heuristic: if the torso's z-velocity is positive and it's above a certain threshold
    # we could consider it's attempting to jump
    jump_initiated_reward = 0
    if velocity_z_coord_of_torso > 0.1 and z_coord_of_torso > 1.0:  # these thresholds are arbitrary and need tuning
        jump_initiated_reward = 1

    # Reward for landing safely, could be as simple as not ending the episode
    safe_landing_reward = 0
    if not done:  # if the humanoid is still alive after the jump
        safe_landing_reward = 1

    jump_reward = jump_initiated_reward + safe_landing_reward
    return jump_reward

def compute_policy_gradient_loss(log_probs, rewards):
    """
    Calculate the policy gradient loss
    :param log_probs: Log probabilities of the actions taken
    :param rewards: Rewards received for each action
    :return: Policy gradient loss
    """
    # We need to calculate the cumulative rewards
    discounted_rewards = []
    total_reward = 0
    for reward in rewards[::-1]:  # reverse rewards list
        total_reward = reward + discount_factor * total_reward
        discounted_rewards.insert(0, total_reward)  # insert at the beginning
    
    # Normalize the rewards
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + eps)

    # Combine the log probabilities with rewards
    policy_gradient = []
    for log_prob, reward in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * reward)  # Negative for gradient ascent

    return torch.stack(policy_gradient).sum()



# Define the RNN architecture
class RNNPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNPolicy, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)
        return x, hidden

# Initialize environment
env = gym.make('Humanoid-v4')

# Define hyperparameters
input_size = env.observation_space.shape[0]
hidden_size = 256
output_size = env.action_space.shape[0]
learning_rate = 1e-3

# Instantiate the RNN policy
rnn_policy = RNNPolicy(input_size, hidden_size, output_size)

# Define optimizer
optimizer = optim.Adam(rnn_policy.parameters(), lr=learning_rate)

# Training loop
num_episodes = 1000
"""
for episode in range(num_episodes):
    observation = env.reset()
    hidden_state = None

    done = False
    total_reward = 0
    
    while not done:
        # Format observation for RNN input
        x = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Get action from RNN policy
        action, hidden_state = rnn_policy(x, hidden_state)
        log_prob = action.log_prob()  # Log probability required for policy gradient
        
        # Apply action
        observation, reward, done, info = env.step(action.detach().numpy().squeeze(0))
        
        total_reward += reward
        
        # Artificially create a "jump" goal for training purposes
        # In practice, you would train the model with a proper RL algorithm
        jump_reward = compute_jump_reward(observation, info)
        total_reward += jump_reward
        
        
        # Your training update would go here
        # This may involve calculating the loss function
        # and using it to update the RNN's parameters
        optimizer.zero_grad()
        loss = compute_policy_gradient_loss(log_probs, rewards)
        loss.backward()
        optimizer.step()
    
    # Print out the performance every few episodes
    if episode % 10 == 0:
        print(f'Episode {episode}: Total Reward: {total_reward}')
"""

# ... [previous code] ...

for episode in range(num_episodes):
    observation = env.reset()
    hidden_state = None

    done = False
    total_reward = 0
    log_probs = []
    rewards = []

    while not done:
        observation_array = np.array(observation)
        x = torch.tensor(observation_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        action_values, hidden_state = rnn_policy(x, hidden_state)
        
        # Apply an activation function and scale to the action space
        # For example, action = torch.tanh(action_values)
        action = torch.tanh(action_values)

        # Convert action to numpy array and apply to the environment
        observation, reward, done, info = env.step(action.detach().numpy().squeeze(0))
        total_reward += reward

        # Compute additional rewards and store them
        jump_reward = compute_jump_reward(observation, info)
        total_reward += jump_reward

        # Store log probabilities and rewards for policy gradient update
        log_prob = action.log_prob()
        log_probs.append(log_prob)
        rewards.append(reward)

        optimizer.zero_grad()
        loss = compute_policy_gradient_loss(log_probs, rewards)
        loss.backward()
        optimizer.step()
    
    if episode % 10 == 0:
        print(f'Episode {episode}: Total Reward: {total_reward}')
