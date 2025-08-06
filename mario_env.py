# Gym is an OpenAI toolkit for RL
import gym
from mario_env_wrapper import SkipFrame, GrayScaleObservation, ResizeObservation
from gym.wrappers import FrameStack

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

class MarioEnv():
    def __init__(self, skip=4, stack=4, grayscale=True, resize_shape=(84, 84), render_mode='rgb'):
        # Initialize Super Mario environment (in v0.26 change render mode to 'human' to see results on the screen)
        if gym.__version__ < '0.26':
            self.env = gym_super_mario_bros.make("SuperMarioBros-v0", new_step_api=True)
        else:
            self.env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode=render_mode, apply_api_compatibility=True)
        self.env = SkipFrame(self.env, skip)
        
        if grayscale:
            self.env = GrayScaleObservation(self.env)
        
        if resize_shape:
            self.env = ResizeObservation(self.env, resize_shape)

        self.env = FrameStack(self.env, num_stack=stack)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
    
if __name__ == "__main__":
    # Example usage
    env = MarioEnv(skip=4, grayscale=True, resize_shape=(84, 84), render_mode='rgb')
    obs = env.reset()
    done = False
    while not done:
        action = env.env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        # print(f"Observation: {obs.shape}, Reward: {reward}, Done: {done}")
        import numpy as np
        obs = np.array(obs)
        import matplotlib.pyplot as plt
        # visualize the last frame in the stack
        plt.imshow(obs[-1], cmap='gray')  # use cmap='gray' for grayscale
        plt.axis('off')
        plt.show()