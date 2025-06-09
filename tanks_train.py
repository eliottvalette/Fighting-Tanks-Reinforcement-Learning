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
from tanks_visualization import TrainingVisualizer

from tanks_paths import TANK_1_WEIGHTS, TANK_2_WEIGHTS, TANK_1_SAVE_WEIGHTS, TANK_2_SAVE_WEIGHTS, RENDERING

# Hyperparameters
EPISODES = 500  # Increased to ensure convergence
GAMMA = 0.99    # Standard discount factor
ALPHA = 0.0005  # Reduced learning rate for stability
GLOBAL_N = 11
MAX_STEPS = 2000  # Round number
EPS_DECAY = 0.99  # Slower decay for better exploration
STATE_SIZE = 24 # +1 for Value

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to run a single episode
def run_episode(agent_1 : TanksAgent, agent_2 : NoBrainBot, epsilon, rendering, episode, render_every, visualizer=None):
    env = TanksGame(max_steps=MAX_STEPS)
    env.reset()
    done = False
    total_reward_1, total_reward_2 = 0, 0
    
    # For collecting action distributions
    episode_actions = []
    episode_values = []

    while not done:
        # Agent 1
        state_1 = env.get_state(num_tank=1)
        actions_1 = agent_1.get_action(state=state_1, epsilon=epsilon)
        next_state_1, reward_1, done, _ = env.step(actions_1, num_tank=1)
        agent_1.remember(state_1, actions_1, reward_1, next_state_1, done)

        # Collect actions for visualization
        episode_actions.append(actions_1)
        
        # Get value predictions for visualization
        state_tensor = torch.FloatTensor(state_1).unsqueeze(0)
        with torch.no_grad():
            _, value = agent_1.model(state_tensor)
            episode_values.append(value.item())

        # Agent 2
        state_2 = env.get_state(num_tank=2)
        actions_2 = agent_2.get_action(state_2)
        _, _, done, _ = env.step(actions_2, num_tank=2)

        total_reward_1 += reward_1

        if rendering and (episode % render_every == 0):
            env.render(rendering=True, clock=2000, epsilon=epsilon)

    # Train the agent
    losses = agent_1.train_model()
    
    # Record metrics if visualizer is provided
    if visualizer:
        visualizer.record_episode(episode, total_reward_1, env.current_step, epsilon, losses)
        visualizer.record_actions_and_values(episode_actions, episode_values)

    return total_reward_1, total_reward_2, env.current_step

# Main Training Loop
def main_training_loop(agent_1, agent_2, episodes, rendering, render_every=10):
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    rewards_history = []
    
    for episode in range(episodes):
        epsilon = np.clip(0.5 * EPS_DECAY ** episode, 0.01, 0.5)
        
        total_reward_1, total_reward_2, steps = run_episode(
            agent_1, agent_2, epsilon, rendering, episode, render_every, visualizer
        )
        
        rewards_history.append(total_reward_1)
        
        print(f'Episode: {episode + 1}, Total Reward Agent 1: {total_reward_1:.2f}, Total Reward Agent 2: {total_reward_2:.2f}, Steps: {steps}, Randomness: {epsilon:.2%}')

        # Save the trained models and generate visualizations every 30 episodes
        if episode % 30 == 29:
            torch.save(agent_1.model.state_dict(), TANK_1_SAVE_WEIGHTS + f"_epoch_{episode+1}.pth")
            
            # Generate visualizations every 30 episodes
            visualizer.generate_all_plots()
    
    # Generate final visualizations
    visualizer.generate_all_plots()
    
    # Print training summary
    stats = visualizer.get_summary_statistics()
    print("\nTraining Summary:")
    for key, value in stats.items():
        print(f"{key}: {value}")


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
    main_training_loop(agent_1, agent_2, episodes=EPISODES, rendering=RENDERING, render_every=1)