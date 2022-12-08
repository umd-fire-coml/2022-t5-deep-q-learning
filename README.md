# 2022-t5-deep-q-learning: Deep Q-Network Model with Atari Games

Reinforcement learning is an area of machine learning that is focused on training agents to take certain actions at certain states in an environment to maximize rewards.
DQN is a reinforcement learning algorithm where a deep learning model is built to find the actions an agent can take at each state. This project trains a DQN model to play a variety of Atari Games, including Q\*bert. It includes a random agent, which generates gameplay based on the machine making random actions, as well as a trained model that attempts to make desired actions to win the game. 

## Demo App
[This is a link to a Google Colab Notebook with the code for this repository](https://colab.research.google.com/drive/1ytTTBTJVBIkCO1YBOC7O50e3x_ECDtNc?usp=sharing)

## Videos
### Process Video Showing How to Use Our Product
[![Watch the video here](https://img.youtube.com/vi/i4_qbyYgTAg/maxresdefault.jpg)](https://youtu.be/i4_qbyYgTAg)
### Gifs of the various models
#### Random Agent
![Random Agent](videos/Random_Agent.gif)
#### Trained
![Trained](videos/Trained.gif)
#### Attempted Optimization and Edits
![](videos/Group_Customized_Rewards.gif)

## Architectures
### Model Architecture
![Model](Architectures/model-architecture.png)
### System Architecture
![System](Architectures/system-architecture.png)

## Directory Guide
- Architectures: The two architecture images above
- Random Agent
  - DQN_Tutorial_Varun.ipynb: The code for running the Random Agent on various Atari Games
- ckpts: contains the ROM for the Atari Games as well as the checkpoints for all of our trained models on the various games
- videos: The videos and gifs above, as well as the product videos for 3 different atari games (Beam Rider, Pong, and Q\*bert)
- DQN.ipynb: Our main code file. This file trains the DQN model, tests it, and produces a gameplay video
- Readme.md: This File
- play.py, train.py, and tt.py: Imported files from the github listed below. These are used as modules in our main DQN code file.

## Instructions
### Random Agent
#### Setup
1. Mount the google drive account
2. Make a directory inside google drive and clone the DQN Atari repository
3. Install some packages to use later, to work with gym and Atari environments
#### Training
1. Import and install the gym version 0.10
2. Create a gym environment for the pong game and reset it
3. Render a single screen of the pong game to show
4. Display the possible actions
5. Perform random action and display the new state, reward, and if the game has ended 
6. Do this multiple times with run_random_agent
7. Run this method on the pong game, display the average reward
#### Testing:
1. Create a video with the random agent in the pong game
2. Generate a video called pong.mp4 which will be in google drive
3. Render the video
4. We can use this on any other atari environment, such as Qbert
5. Repeat this process with the new environment and render a video of the gameplay
### Trained Model
#### Setup
1. Set the runtime to GPU and mount the drive with correct work directory
2. Check if the python is 3.6 or 3.7
3. Check if the tensorflow version is 1.15 or earlier
4. Check if the gym-atari version is 0.10 and the gym-rom-license is installed
5. Check if the pyglet version is 1.3
6. Implement functions from DQN random agent: run_random_agent, create_video, render_video, run_custom_environment
#### Training
1. Modify test.py file. By adding an additional argument, we are able to change the default args in test_params. Same for train_params and play_params.
2. Create the DQN by initializing the model and setting environment and StateBuffer. The environment will be created through parameter env and the state will be passed into the CNN for DQN model. The function will return a DQN and input state placeholder.
3. Create a tensorflow session that will run the model and load the checkpoints.
4. By loading the checkpoints to the model, we are able to run the trained agent in the environment and save the corresponding screens and rewards. For each episode, the agent will perform the action calculated by the DQN model. But in the early steps, the agent will still go with a random action.
#### Testing
1. The function evaluate_model will be used to compare average reward from each episode.
2. Organize and merge all the functions into train_custom_environment(env) and test_custom_environment(env). The first one will be used to run DQN, training the agent and saving the checkpoints. The second will be used to generate a video and compare the result.
### Additional Notes
- network.py defines the architecture, loss function, and optimizer for the DQN Model.
- utils.py defines how the data is being preprocessed.

## Citations
1. msinto93. 2018. MSINTO93/DQN_ATARI: A tensorflow implementation of a Deep Q Network (DQN) for playing Atari Games. (October 2018). Retrieved December 8, 2022 from https://github.com/msinto93/DQN_Atari 
2. Volodymyr Mnih et al. 2013. Playing Atari with deep reinforcement learning. (December 2013). Retrieved December 8, 2022 from https://arxiv.org/abs/1312.5602 
