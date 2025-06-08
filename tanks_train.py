# tanks_train.py
import numpy as np
import random as rd
import pygame
import torch
import time
from tanks_agent import TanksAgent
from tanks_no_brain_bot import NoBrainBot
from tanks_game import TanksGame
import matplotlib.pyplot as plt

from tanks_paths import TANK_1_WEIGHTS, TANK_2_WEIGHTS, TANK_1_SAVE_WEIGHTS, TANK_2_SAVE_WEIGHTS, RENDERING

# Hyperparameters
EPISODES = 400
GAMMA = 0.9985
ALPHA = 0.001
GLOBAL_N = 11
MAX_STEPS = 1_998 
EPS_DECAY = 0.98
STATE_SIZE = 23

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to run a single episode
# tanks_train.py
def run_episode(agent_1 : TanksAgent, agent_2 : NoBrainBot, epsilon, rendering, episode, render_every):
    env = TanksGame(max_steps=MAX_STEPS)
    env.reset()
    done = False
    total_reward_1, total_reward_2 = 0, 0

    while not done:
        # Agent 1
        state_1 = env.get_state(num_tank=1)
        actions_1 = agent_1.get_action(state=state_1, epsilon=epsilon)
        next_state_1, reward_1, done, _ = env.step(actions_1, num_tank=1)
        agent_1.remember(state_1, actions_1, reward_1, next_state_1, done)

        # Agent 2
        state_2 = env.get_state(num_tank=2)
        actions_2 = agent_2.get_action(state_2)
        _, _, done, _ = env.step(actions_2, num_tank=2)

        total_reward_1 += reward_1

        env.render(rendering=True, clock = 2000, epsilon=epsilon)

    agent_1.train_model()

    return total_reward_1, total_reward_2, env.current_step  # Fix attribute name

# Main Training Loop
def main_training_loop(agent_1, agent_2, episodes, rendering, render_every = 10):
    for episode in range(episodes):
        epsilon = np.clip(0.5 * EPS_DECAY ** episode, 0.01, 0.5)
        
        total_reward_1, total_reward_2, steps = run_episode(agent_1, agent_2, epsilon, rendering, episode, render_every)

        print(f'Episode: {episode + 1}, Total Reward Agent 1: {total_reward_1:.2f}, Total Reward Agent 2: {total_reward_2:.2f}, Steps: {steps}, Randomness: {epsilon:.2%}')

        # Save the trained models every 50 episodes
        if episode % 50 == 49:
            torch.save(agent_1.model.state_dict(), TANK_1_SAVE_WEIGHTS + f"_epoch_{episode+1}.pth")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Create the Q-learning agent
    agent_1 = TanksAgent(
        state_size=STATE_SIZE,
        action_sizes=[3, 3, 3, 2], # [move, rotate, strafe, fire]
        gamma = GAMMA,
        learning_rate = ALPHA,
        load_model = False,
    )

    '''
    agent_2 = NoBrainBot(
        state_size=STATE_SIZE,
        action_sizes=[3, 3, 3, 2], # [move, rotate, strafe, fire]
        gamma = GAMMA,
        learning_rate = ALPHA,
        load_model = False,
    )
    '''

    agent_2 = NoBrainBot(
        state_size=STATE_SIZE,
        action_sizes=[3, 3, 3, 2]
    )

    if agent_1.load_model:
        print("Loading model 1 weights...")
        agent_1.model.load_state_dict(torch.load(TANK_1_WEIGHTS, weights_only=True))

    # Start the training loop
    main_training_loop(agent_1, agent_2, episodes = EPISODES, rendering = RENDERING, render_every = 1)