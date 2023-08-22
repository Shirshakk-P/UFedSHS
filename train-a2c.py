import os
import gym 
import tensorboard
import stable_baselines3
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

import numpy as np
from UAVE import UAVEnvironment

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_dir = "a2c_logs/"
os.makedirs(log_dir, exist_ok=True)
env1 = UAVEnvironment(5, 12, 1e5)

env = Monitor(env1, log_dir)


print("A2C Running...")
model=A2C("MlpPolicy", env1, verbose=1, tensorboard_log="./tb_logs/")
model.learn(total_timesteps=1e5, tb_log_name="A2Cnew_RUN512_100k", progress_bar=True)
model.save("A2Cmodel")
