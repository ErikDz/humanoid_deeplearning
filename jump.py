import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
from gymnasium import RewardWrapper
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



from gym import RewardWrapper
env_name = 'Humanoid-v4'
env = gym.make(env_name, render_mode=None)

def jump_reward(reward):
    #print(f"Initial reward {reward}")
    torso=env.state_vector()[2] #position of torso
    z_orientation=env.state_vector()[2+3]
    z_velocity=env.state_vector()[2+24]
    #print("State vector")
    np.set_printoptions(precision=6, suppress=True)
    #print(env.state_vector())
    salto=0
    #print(f"pos torso {torso:.3f}")
    if (torso > 1.5):
        salto+=10
    else:
        salto-=10
    if (-0.18 <= z_orientation <= 0.18):
        salto+=10
    else:
        salto-=10
    if (z_velocity > 0):
        salto+=10
    else:
        salto-=10

    default_rewards= env.healthy_reward #Siguiendo la fórmula de arriba accediendo a los atributos básidos
    #print(f"Default {default_rewards}")
    #print(f"Salto {salto}")
    reward=reward + salto + default_rewards
    #print(f"Final {reward}")
    #print(env.healthy_reward)
    return reward


env = gym.wrappers.TransformReward(env, jump_reward)
env.reset()

# Rest of your code remains the same
if __name__ == '__main__':
    gymenv_name = 'Humanoid-v4'
    sb3_algo = 'SAC'
    path_to_model = 'path_to_your_model_file'

    train_model = False
    test_model = True

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

