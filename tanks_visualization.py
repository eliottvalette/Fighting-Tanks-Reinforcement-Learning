import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import os
from datetime import datetime
from collections import defaultdict
import json
import pickle

class TrainingVisualizer:
    def __init__(self, save_dir_data="visualization_data", save_dir_png="visualization_png"):
        """Initialize the training visualizer with a directory to save data."""
        self.save_dir_data = save_dir_data
        os.makedirs(save_dir_data, exist_ok=True)
        self.save_dir_png = save_dir_png
        os.makedirs(save_dir_png, exist_ok=True)
        
        # Initialize data structures to store metrics
        self.rewards_history = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.losses_agent1 = defaultdict(list)  # {loss_name: [values]} for agent 1
        self.losses_agent2 = defaultdict(list)  # {loss_name: [values]} for agent 2
        self.action_distributions = []
        self.value_distributions = []
        
        # New metrics for enhanced tracking
        self.additional_metrics = defaultdict(list)  # For custom metrics like hit accuracy
        
        # Create a consolidated metrics file that will be updated after each episode
        self.metrics_file = os.path.join(save_dir_data, "all_metrics.json")
        self.backup_metrics_file = os.path.join(save_dir_data, "metrics_backup.json")
        self.backup_actions_file = os.path.join(save_dir_data, "actions_backup.pkl")
        self.backup_values_file = os.path.join(save_dir_data, "values_backup.pkl")
        
    def record_episode(self, episode, reward, steps, epsilon, losses_1, losses_2):
        """Record metrics for a completed episode."""
        self.rewards_history.append(reward)
        self.episode_lengths.append(steps)
        self.epsilon_history.append(epsilon)
        
        # Record losses for agent 1
        if losses_1:
            for loss_name, loss_value in losses_1.items():
                self.losses_agent1[loss_name].append(loss_value)
        
        # Record losses for agent 2
        if losses_2:
            for loss_name, loss_value in losses_2.items():
                self.losses_agent2[loss_name].append(loss_value)
                
        # Save all metrics after every episode to ensure complete data
        self.save_metrics()
            
    def record_additional_metrics(self, metrics_dict):
        """Record additional custom metrics."""
        for metric_name, value in metrics_dict.items():
            self.additional_metrics[metric_name].append(value)
            
    def record_actions_and_values(self, actions, values):
        """Record action and value distributions for analysis."""
        self.action_distributions.append(actions)
        self.value_distributions.append(values)
        
    def save_metrics(self):
        """Save all metrics to disk, always saving the complete history."""
        metrics = {
            'rewards': self.rewards_history,
            'episode_lengths': self.episode_lengths,
            'epsilon': self.epsilon_history,
            'losses_agent1': dict(self.losses_agent1),
            'losses_agent2': dict(self.losses_agent2),
            'additional_metrics': dict(self.additional_metrics)
        }
        
        # Save to the consolidated file (overwriting previous version)
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f)
            
        # Every 10 episodes, also create a backup (overwriting previous backup)
        current_episode = len(self.rewards_history)
        if current_episode % 10 == 0:
            # Save metrics backup (overwriting previous backup)
            with open(self.backup_metrics_file, 'w') as f:
                json.dump(metrics, f)
            
            # Use pickle to save action distributions and values (overwriting previous backups)
            if self.action_distributions:
                with open(self.backup_actions_file, 'wb') as f:
                    pickle.dump(self.action_distributions, f)
                    
            if self.value_distributions:
                with open(self.backup_values_file, 'wb') as f:
                    pickle.dump(self.value_distributions, f)
            
    def load_metrics(self, filename=None):
        """Load metrics from a JSON file."""
        if filename is None:
            filename = self.metrics_file
            
        if not os.path.exists(filename):
            return {}
            
        with open(filename, 'r') as f:
            metrics = json.load(f)
        return metrics
    
    def plot_training_curve(self, window_size=10):
        """Plot the reward curve with a moving average."""
        rewards = np.array(self.rewards_history)
        episodes = np.arange(len(rewards))
        
        # Check if we have enough data for moving average
        if len(rewards) < window_size:
            # If not enough data, just plot raw rewards
            plt.figure(figsize=(12, 6))
            plt.plot(episodes, rewards, alpha=0.7, color='blue', label='Raw rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Reward Curve (Insufficient data for moving average)')
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            # Calculate moving average only if we have enough data
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            
            plt.figure(figsize=(12, 6))
            plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw rewards')
            plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 
                    color='blue', linewidth=2, label=f'{window_size}-episode moving average')
            
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Reward Curve')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Save the figure
        plt.savefig(os.path.join(self.save_dir_png, 'reward_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_loss_curves(self):
        """Create two separate figures for actor and critic losses of each agent."""
        if not self.losses_agent1 and not self.losses_agent2:
            print("No loss data available")
            return

        # Check if we have the required loss data
        if 'actor_loss' not in self.losses_agent1 or 'actor_loss' not in self.losses_agent2 or \
           'critic_loss' not in self.losses_agent1 or 'critic_loss' not in self.losses_agent2:
            print("Missing required loss data (actor_loss or critic_loss)")
            return

        # Check if loss arrays are not empty
        if len(self.losses_agent1['actor_loss']) == 0 or len(self.losses_agent2['actor_loss']) == 0 or \
           len(self.losses_agent1['critic_loss']) == 0 or len(self.losses_agent2['critic_loss']) == 0:
            print("Loss arrays are empty")
            return

        # Set up color and style schemes
        colors = ['#003049', '#006DAA', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']

        # First figure: Actor losses
        plt.figure(figsize=(12, 6))
        color_1 = colors[0]
        color_2 = colors[-1]
        plt.plot(self.losses_agent1['actor_loss'], color=color_1, linewidth=2, label='Actor Loss Agent 1')
        plt.plot(self.losses_agent2['actor_loss'], color=color_2, linewidth=2, label='Actor Loss Agent 2')
        plt.title('Actor Losses of Both Agents', fontsize=16, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=14)
        plt.ylabel('Loss Value', fontsize=14)
        plt.grid(True, alpha=0.5)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc='best', fontsize=12, framealpha=0.7, fancybox=True, shadow=True)
        plt.tight_layout()
        
        # Save the first figure
        plt.savefig(os.path.join(self.save_dir_png, 'actor_loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Second figure: Critic losses
        plt.figure(figsize=(12, 6))
        color_1 = colors[1]
        color_2 = colors[-2]
        plt.plot(self.losses_agent1['critic_loss'], color=color_1, linewidth=2, label='Critic Loss Agent 1')
        plt.plot(self.losses_agent2['critic_loss'], color=color_2, linewidth=2, label='Critic Loss Agent 2')
        plt.title('Critic Losses of Both Agents', fontsize=16, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=14)
        plt.ylabel('Loss Value', fontsize=14)
        plt.grid(True, alpha=0.5)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc='best', fontsize=12, framealpha=0.7, fancybox=True, shadow=True)
        plt.tight_layout()
        
        # Save the second figure
        plt.savefig(os.path.join(self.save_dir_png, 'critic_loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


    def plot_reward_histogram(self, bins=20):
        """Plot histogram of rewards to analyze distribution."""
        if not self.rewards_history:
            print("No reward data available for histogram")
            return
            
        plt.figure(figsize=(10, 6))
        
        sns.histplot(self.rewards_history, bins=bins, kde=True)
        plt.axvline(np.mean(self.rewards_history), color='red', linestyle='dashed', 
                   linewidth=2, label=f'Mean: {np.mean(self.rewards_history):.2f}')
        
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.legend()
        
        # Save the figure
        plt.savefig(os.path.join(self.save_dir_png, 'reward_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_action_distributions(self):
        """Plot the distribution of actions taken over time."""
        if not self.action_distributions:
            print("No action distribution data available")
            return
            
        # Count action occurrences across all episodes
        action_counts = {
            0: [0, 0, 0],  # Move: [forward, backward, none]
            1: [0, 0, 0],  # Rotate: [right, left, none]
            2: [0, 0, 0],  # Strafe: [left, right, none]
            3: [0, 0]      # Fire: [fire, don't fire]
        }
        
        # Process all episodes' actions
        for episode_actions in self.action_distributions:
            for action in episode_actions:
                # Count each action type
                for i, action_value in enumerate(action):
                    if i == 3:  # Fire action has only 2 possible values
                        action_counts[i][action_value] += 1
                    else:
                        action_counts[i][action_value] += 1
                        
        # Plot distributions
        action_types = ["Move", "Rotate", "Strafe", "Fire"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, action_type in enumerate(action_types):
            counts = action_counts[i]
            if i < 3:
                labels = ['Forward/Right', 'Backward/Left', 'No Action']
            else:
                labels = ['Fire', 'No Fire']
                
            # Plot pie chart
            axes[i].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[i].set_title(f'{action_type} Action Distribution')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir_png, 'action_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_value_distribution(self):
        """Plot the distribution of predicted state values."""
        if not self.value_distributions:
            print("No value distribution data available")
            return
            
        # Flatten all value predictions
        all_values = []
        for episode_values in self.value_distributions:
            all_values.extend(episode_values)
        
        if not all_values:
            print("No value data available for distribution plot")
            return
            
        plt.figure(figsize=(10, 6))
        sns.histplot(all_values, bins=50, kde=True)
        plt.axvline(np.mean(all_values), color='red', linestyle='dashed', 
                   linewidth=2, label=f'Mean: {np.mean(all_values):.2f}')
        
        plt.xlabel('Predicted State Value')
        plt.ylabel('Frequency')
        plt.title('Value Function Distribution')
        plt.legend()
        
        # Save the figure
        plt.savefig(os.path.join(self.save_dir_png, 'value_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_epsilon_decay(self):
        """Plot the exploration rate (epsilon) over time."""
        if not self.epsilon_history:
            print("No epsilon data available")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.epsilon_history)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon (Exploration Rate)')
        plt.title('Exploration Rate Decay')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Save the figure
        plt.savefig(os.path.join(self.save_dir_png, 'epsilon_decay.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_episode_lengths(self):
        """Plot episode lengths over time."""
        if not self.episode_lengths:
            print("No episode length data available")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Length Over Time')
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.savefig(os.path.join(self.save_dir_png, 'episode_lengths.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_additional_metrics(self):
        """Plot additional tracked metrics over time."""
        if not self.additional_metrics:
            return
        
        num_metrics = len(self.additional_metrics)
        if num_metrics > 0:
            fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4*num_metrics))
            if num_metrics == 1:
                axes = [axes]
                
            for i, (metric_name, values) in enumerate(self.additional_metrics.items()):
                axes[i].plot(values)
                axes[i].set_title(f'{metric_name.replace("_", " ").title()} over Episodes')
                axes[i].set_xlabel('Episode')
                axes[i].set_ylabel(metric_name.replace("_", " ").title())
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir_png, 'additional_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
    def generate_all_plots(self):
        """Generate all available plots."""
        self.plot_training_curve()
        self.plot_loss_curves()
        self.plot_reward_histogram()
        self.plot_action_distributions()
        self.plot_value_distribution()
        self.plot_epsilon_decay()
        self.plot_episode_lengths()
        self.plot_additional_metrics()
        
    def get_summary_statistics(self):
        """Calculate and return summary statistics of the training."""
        stats = {
            'total_episodes': len(self.rewards_history),
            'mean_reward': np.mean(self.rewards_history) if self.rewards_history else None,
            'max_reward': np.max(self.rewards_history) if self.rewards_history else None,
            'min_reward': np.min(self.rewards_history) if self.rewards_history else None,
            'std_reward': np.std(self.rewards_history) if self.rewards_history else None,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else None,
        }
        
        # Add agent 1 loss statistics if available
        for loss_name, values in self.losses_agent1.items():
            if values:
                stats[f'agent1_mean_{loss_name}'] = np.mean(values)
                stats[f'agent1_last_{loss_name}'] = values[-1] if values else None
        
        # Add agent 2 loss statistics if available
        for loss_name, values in self.losses_agent2.items():
            if values:
                stats[f'agent2_mean_{loss_name}'] = np.mean(values)
                stats[f'agent2_last_{loss_name}'] = values[-1] if values else None
                
        # Add additional metrics if available
        for metric_name, values in self.additional_metrics.items():
            if values:
                stats[f'mean_{metric_name}'] = np.mean(values)
                stats[f'last_{metric_name}'] = values[-1] if values else None
                
        return stats
        
    def create_learning_curve_comparison(self, other_metrics_files, labels):
        """Compare learning curves from different training runs."""
        plt.figure(figsize=(12, 8))
        
        # Plot current metrics
        window_size = 10
        rewards = np.array(self.rewards_history)
        if len(rewards) > 0:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 
                    linewidth=2, label='Current run')
        
        # Plot other metrics
        for i, filename in enumerate(other_metrics_files):
            metrics = self.load_metrics(filename)
            rewards = np.array(metrics['rewards'])
            if len(rewards) > 0:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 
                        linewidth=2, label=labels[i] if i < len(labels) else f'Run {i+1}')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward (Moving Average)')
        plt.title('Learning Curve Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the figure
        plt.savefig(os.path.join(self.save_dir_png, 'learning_curve_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Helper function to extract model insights
def analyze_model(actor, critic, state_size):
    """Analyze the trained model's weights and biases."""
    model_stats = {}
    
    # Generate a random batch of states to analyze model behavior
    test_states = torch.randn(100, state_size)
    with torch.no_grad():
        action_probs = actor(test_states)
        values = critic(test_states)
    
    # Analyze action probabilities
    action_stats = []
    for i, probs in enumerate(action_probs):
        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1))
        max_prob = torch.mean(torch.max(probs, dim=1)[0])
        action_stats.append({
            'action_type': i,
            'mean_entropy': entropy.item(),
            'mean_max_prob': max_prob.item()
        })
    
    model_stats['action_stats'] = action_stats
    model_stats['mean_value'] = values.mean().item()
    model_stats['value_std'] = values.std().item()
    
    return model_stats

# Example usage:
if __name__ == "__main__":
    # This would typically be in the main training script
    visualizer = TrainingVisualizer()
    
    # After training is complete:
    visualizer.generate_all_plots()
    
    # Print summary statistics
    stats = visualizer.get_summary_statistics()
    print("Training Summary:")
    for key, value in stats.items():
        print(f"{key}: {value}") 