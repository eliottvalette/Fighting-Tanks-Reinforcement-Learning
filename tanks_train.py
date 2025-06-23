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
EPISODES = 5_000  # Increased to ensure convergence
GAMMA = 0.99    # Standard discount factor
ALPHA = 0.0003  # Increased learning rate for faster learning
GLOBAL_N = 11
MAX_STEPS = 2000  # Round number
EPS_DECAY = 0.99  # Slower decay for better exploration
STATE_SIZE = 30 + 1 # +1 for Value
SHORT_MEMORY_SIZE = MAX_STEPS
LONG_MEMORY_SIZE = 10000
LONG_MEMORY_UPDATE_FREQUENCY = 100
OFF_POLICY_TRAINING = False

def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to run a single episode
def run_episode(agent_1 : TanksAgent, agent_2 : TanksAgent, epsilon, rendering, episode, render_every, visualizer : TrainingVisualizer):
    env = TanksGame(max_steps=MAX_STEPS)
    env.reset()
    done = False
    total_reward_1, total_reward_2 = 0, 0
    
    # For collecting action distributions
    episode_actions = []
    episode_values = []
    step_count = 0
    long_memory_episode_delay = rd.randint(0, LONG_MEMORY_UPDATE_FREQUENCY) # So that it's not always the same episode (+ periodic update) that is saved

    print(f'long_memory_length : {len(agent_1.long_memory)}, short_memory_length : {len(agent_1.short_memory)}')

    while not done and step_count < MAX_STEPS:
        step_count += 1
        # Agent 1
        state_1 = env.get_state(num_tank=1)
        actions_1 = agent_1.get_action(state=state_1, epsilon=epsilon, action_sizes=agent_1.action_sizes)
        next_state_1, reward_1, done, _ = env.step(actions_1, num_tank=1)
        agent_1.remember_short(state_1, actions_1, reward_1, next_state_1, done)
        if (step_count + long_memory_episode_delay) % LONG_MEMORY_UPDATE_FREQUENCY == 0 and OFF_POLICY_TRAINING:
            agent_1.remember_long(state_1, actions_1, reward_1, next_state_1, done)

        # Collect actions for visualization
        episode_actions.append(actions_1)
        
        # Get value predictions for visualization
        state_tensor = torch.FloatTensor(state_1).unsqueeze(0)
        agent_1.critic.eval()  # Set critic model to evaluation mode for inference
        with torch.no_grad():
            _, value = agent_1.critic(state_tensor)
            episode_values.append(value.item())
        agent_1.critic.train()  # Set critic model back to training mode

        # Agent 2
        state_2 = env.get_state(num_tank=2)
        actions_2 = agent_2.get_action(state_2, epsilon=epsilon, action_sizes=agent_2.action_sizes)
        next_state_2, reward_2, done, _ = env.step(actions_2, num_tank=2)
        agent_2.remember_short(state_2, actions_2, reward_2, next_state_2, done)
        if (step_count + long_memory_episode_delay) % LONG_MEMORY_UPDATE_FREQUENCY == 0 and OFF_POLICY_TRAINING:
            agent_2.remember_long(state_2, actions_2, reward_2, next_state_2, done)

        # Collect actions for visualization
        episode_actions.append(actions_2)
        
        # Get value predictions for visualization
        state_tensor = torch.FloatTensor(state_2).unsqueeze(0)
        agent_2.critic.eval()  # Set critic model to evaluation mode for inference
        with torch.no_grad():
            _, value = agent_2.critic(state_tensor)
            episode_values.append(value.item())
        agent_2.critic.train()  # Set critic model back to training mode

        total_reward_1 += reward_1
        total_reward_2 += reward_2

        # Train the agent with online single-step updates
        agent_1.train_model_batch(batch_size=64, short_memory = True)
        agent_2.train_model_batch(batch_size=64, short_memory = True)

        if rendering and (episode % render_every == 0):
            env.render(rendering=True, clock=60, epsilon=epsilon)  # Reduced clock speed for better visualization

    # Additional batch training at the end of the episode
    losses_1 = agent_1.train_model_batch(batch_size=32, short_memory = not OFF_POLICY_TRAINING)
    losses_2 = agent_2.train_model_batch(batch_size=32, short_memory = not OFF_POLICY_TRAINING)
    
    # Record metrics if visualizer is provided
    if visualizer:
        visualizer.record_episode(episode, total_reward_1, step_count, epsilon, losses_1, losses_2)
        visualizer.record_actions_and_values(episode_actions, episode_values)

    return total_reward_1, total_reward_2, step_count

# Main Training Loop
def main_training_loop(agent_1, agent_2, episodes, rendering, render_every=10):
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    rewards_history = []
    
    try:
        for episode in range(episodes):
            epsilon = max(0.05, 0.5 * (EPS_DECAY ** episode))  # Better epsilon annealing schedule
            
            total_reward_1, total_reward_2, steps = run_episode(
                agent_1, agent_2, epsilon, rendering, episode, render_every, visualizer
            )
            
            rewards_history.append(total_reward_1)
            
            print(f'Episode: {episode + 1}, Total Reward Agent 1: {total_reward_1:.2f}, Steps: {steps}, Randomness: {epsilon:.2%}')

            # Learning rate decay
            if episode > 0 and episode % 100 == 0:
                agent_1.adjust_learning_rate(0.95)  # Reduce learning rate by 5% every 100 episodes
                agent_2.adjust_learning_rate(0.95)  # Reduce learning rate by 5% every 100 episodes

            # Save the trained models and generate visualizations every 30 episodes
            if episode % 10 == 9:
                agent_1.save(TANK_1_SAVE_WEIGHTS + f"_epoch_{episode+1}.pth")
                agent_2.save(TANK_2_SAVE_WEIGHTS + f"_epoch_{episode+1}.pth")
                
                # Generate visualizations every 30 episodes
                visualizer.generate_all_plots()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Generating final plots...")
        visualizer.generate_all_plots()
        print("Final plots generated successfully.")
        return
    
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
        gamma=GAMMA,
        learning_rate=ALPHA,
        entropy_coeff=0.01,  # Decreased for more exploitation
        policy_loss_coeff=1.0,
        load_model=False,
        short_memory_size=SHORT_MEMORY_SIZE,
        long_memory_size=LONG_MEMORY_SIZE,
    )
    # Set agent identity for loading models
    agent_1.is_agent_1 = True

    agent_2 = TanksAgent(
        state_size=STATE_SIZE,
        action_sizes=[3, 3, 3, 2], # [move, rotate, strafe, fire]
        gamma=GAMMA,
        learning_rate=ALPHA,
        entropy_coeff=0.01,  # Decreased for more exploitation
        policy_loss_coeff=1.0,
        load_model=False,
        short_memory_size=SHORT_MEMORY_SIZE,
        long_memory_size=LONG_MEMORY_SIZE,
    )

    # Start the training loop
    main_training_loop(agent_1, agent_2, episodes=EPISODES, rendering=RENDERING, render_every=1)
