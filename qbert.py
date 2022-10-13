import gym
import tensorflow as tf #memory used?
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

env = gym.make("ALE/Qbert-v5")

screens = []

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space
print("Size of Action Space ->  {}".format(num_actions))

def create_video(screens, filename):

  # get the height and width of a single frame
  height, width, _ = screens[0].shape
  # get the fourcc code for mp4 videos
  fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
  # create a video for this environment
  video = cv2.VideoWriter(filename, fourcc, 60, (width, height))
  # write each frame to the videos
  for screen in tqdm(screens):
    # flip RGB to BGR for cv2
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    # write frame
    video.write(screen)
  # save the video and close the writer
  cv2.destroyAllWindows()
  video.release()  

