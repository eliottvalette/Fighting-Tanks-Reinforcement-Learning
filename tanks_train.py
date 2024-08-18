# tanks_train.py
import numpy as np
import random as rd
import multiprocessing as mp

import pygame
import torch
import time
from tanks_agent import TanksAgent
from tanks_game import TanksGame
import matplotlib.pyplot as plt

from tanks_paths import TANK_1_WEIGHTS, TANK_2_WEIGHTS, TANK_1_SAVE_WEIGHTS, TANK_2_SAVE_WEIGHTS, RENDERING

# Hyperparameters
EPISODES = 10_000
GAMMA = 0.99
ALPHA = 0.001
GLOBAL_N = 11
MAX_STEPS = 500 
EPS_DECAY = 0.85
STATE_SIZE = 14

# Function to run a single episode in parallel
def run_episode(agent_1, agent_2, epsilon, rendering, episode=0, steps_before_update=5):
    # Create a new environment inside the process
    env = TanksGame()

    print(f'---Running episode {episode} ---')
    env.reset()
    done = False
    total_reward_1 = 0
    total_reward_2 = 0
    steps = 0

    while not done and steps < MAX_STEPS:
        state_1 = env.get_state(num_tank=1)
        actions_1 = agent_1.get_actions(state_1, epsilon)
        next_state_1, reward_1, done, _ = env.step(actions_1, num_tank=1)

        state_2 = env.get_state(num_tank=2)
        actions_2 = agent_2.get_actions(state_2, epsilon)
        next_state_2, reward_2, done, _ = env.step(actions_2, num_tank=2)
        
        agent_1.train_model(state_1, actions_1, reward_1, next_state_1, done)
        agent_2.train_model(state_2, actions_2, reward_2, next_state_2, done)

        total_reward_1 += reward_1
        total_reward_2 += reward_2

        env.render(rendering=rendering, clock=100, epsilon=epsilon)

        steps += 1

    return total_reward_1, total_reward_2, steps

# Main Training Loop
def main_training_loop(agent_1, agent_2, EPISODES, rendering):
    for episode in range(EPISODES):
        epsilon = np.clip(EPS_DECAY ** episode, 0.01, 0.75)

        # Render this episode for visualization
        total_reward_1, total_reward_2, steps = run_episode(agent_1, agent_2, epsilon, rendering, episode)
        
        agent_1.replay()
        agent_2.replay()

        print(f'Episode: {episode + 1}, Total Reward Agent 1: {total_reward_1}, Total Reward Agent 2: {total_reward_2}, Steps: {steps}')
            
        # Save the trained models every 50 episodes
        if episode % 10 == 9:
            torch.save(agent_1.model.state_dict(), TANK_1_SAVE_WEIGHTS + f"_epoch_{episode+1}.pth")
            torch.save(agent_2.model.state_dict(), TANK_2_SAVE_WEIGHTS + f"_epoch_{episode+1}.pth")


if __name__ == "__main__":
    # Create the Q-learning agent
    agent_1 = TanksAgent(
        state_size=STATE_SIZE,
        action_sizes=[3, 3, 3, 2], # [move, rotate, strafe, fire]
        gamma = GAMMA,
        learning_rate = ALPHA,
        load_model = False,
    )

    agent_2 = TanksAgent(
        state_size=STATE_SIZE,
        action_sizes=[3, 3, 3, 2], # [move, rotate, strafe, fire]
        gamma = GAMMA,
        learning_rate = ALPHA,
        load_model = False,
    )

    if agent_1.load_model:
        print("Loading model 1 weights...")
        agent_1.model.load_state_dict(torch.load(TANK_1_WEIGHTS))

    if agent_2.load_model:
        print("Loading model 2 weights...")
        agent_2.model.load_state_dict(torch.load(TANK_2_WEIGHTS))

    # Start the training loop
    main_training_loop(agent_1, agent_2, EPISODES=100, rendering = RENDERING)