import gym
import tensorflow as tf #memory used?
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import atari_py

print(atari_py.list_games())

env = gym.make("ALE/Qbert-v5")

