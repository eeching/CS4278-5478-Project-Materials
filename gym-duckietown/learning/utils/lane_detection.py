import ast
import argparse
import logging

import os
import numpy as np

# Duckietown Specific
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
  DtRewardWrapper, ActionWrapper, ResizeWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def extract():

  env = launch_env()
  print("Initialized environment")

  # Wrappers
  env = ResizeWrapper(env)
  env = NormalizeWrapper(env)
  env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
  env = ActionWrapper(env)
  env = DtRewardWrapper(env)
  print("Initialized Wrappers")

  state_dim = env.observation_space.shape
  action_dim = env.action_space.shape[0]
  max_action = float(env.action_space.high[0])

  total_timesteps = 0
  timesteps_since_eval = 0
  episode_num = 0
  done = True
  episode_reward = None
  env_counter = 0
  reward = 0
  episode_timesteps = 0
  obs = env.reset()

  print("Starting training")
  while total_timesteps < 2:
    action = env.action_space.sample()
    # Perform action
    new_obs, reward, done, _ = env.step(action)

    obs = new_obs
    print(obs)
    print(action)
    total_timesteps += 1




if __name__ == '__main__':
  extract()
