import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
from gymnasium import Wrapper
import numpy as np


# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env, sb3_algo):
    if sb3_algo == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif sb3_algo == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    else:
        print('Algorithm not found')
        return

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")

def test(env, sb3_algo, path_to_model):
    if sb3_algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif sb3_algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif sb3_algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    else:
        print('Algorithm not found')
        return

    obs = env.reset()
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

        if done:
            extra_steps -= 1
            if extra_steps < 0:
                break
import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np

class JumpRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.jump_reward_weight = 10.0  # Weight for the jump reward
        self.desired_height = 1.5       # Desired height to achieve for jumping

    def step(self, action):
        obs, reward, done, info, extra = self.env.step(action)
        jump_reward = self.calculate_jump_reward(obs)
        reward += jump_reward
        return obs, reward, done, info, extra

    def calculate_jump_reward(self, obs):
        # Assuming the z-coordinate of the torso is the first element in the observation
        torso_height = obs[0]
        if torso_height > self.desired_height:
            return self.jump_reward_weight
        return 0.0

# Rest of your code remains the same
if __name__ == '__main__':
    gymenv_name = 'Humanoid-v4'
    sb3_algo = 'SAC'
    path_to_model = 'path_to_your_model_file'

    train_model = True
    test_model = False

    if train_model:
        gymenv = gym.make(gymenv_name, render_mode=None)
        wrapped_env = JumpRewardWrapper(gymenv)  # Wrap the environment
        train(wrapped_env, sb3_algo)

    if test_model:
        if os.path.isfile(path_to_model):
            gymenv = gym.make(gymenv_name, render_mode='human')
            wrapped_env = JumpRewardWrapper(gymenv)  # Wrap the environment
            test(wrapped_env, sb3_algo, path_to_model=path_to_model)
        else:
            print(f'{path_to_model} not found.')

