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
        
        # Calculate moving average
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
        """Create a double plot (side-by-side subplots) for all losses of each agent."""
        if not self.losses_agent1 and not self.losses_agent2:
            print("No loss data available")
            return

        # Prepare loss types for each agent
        loss_types_agent1 = list(self.losses_agent1.keys())
        loss_types_agent2 = list(self.losses_agent2.keys())

        # Set up color and style schemes
        colors = ['red', 'blue', 'green', 'yellow', 'purple']

        # Create a double plot (side-by-side subplots)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
        agent_titles = ['Agent 1', 'Agent 2']

        # Plot for Agent 1
        ax = axes[0]
        for i, loss_type in enumerate(sorted(loss_types_agent1)):
            color = colors[i % len(colors)]
            ax.plot(self.losses_agent1[loss_type], color=color, linewidth=2, label=loss_type)
            # Add moving average for total_loss if available
            if loss_type == 'total_loss':
                window_size = min(15, len(self.losses_agent1['total_loss']))
                if window_size > 1:
                    moving_avg = np.convolve(self.losses_agent1['total_loss'],
                                            np.ones(window_size)/window_size, mode='valid')
                    ax.plot(np.arange(window_size-1, len(self.losses_agent1['total_loss'])),
                            moving_avg, 'k-', linewidth=3, alpha=0.7,
                            label=f'total_loss (MA{window_size})')
        ax.set_title(f'All Loss Components - {agent_titles[0]}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=14)
        ax.set_ylabel('Loss Value', fontsize=14)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if len(loss_types_agent1) > 4:
            ncol = 2
            loc = 'upper center'
            bbox_to_anchor = (0.5, -0.1)
        else:
            ncol = 1
            loc = 'best'
            bbox_to_anchor = None
        ax.legend(loc=loc, fontsize=12, framealpha=0.7, fancybox=True, shadow=True, ncol=ncol, bbox_to_anchor=bbox_to_anchor)
        ax.set_yscale('symlog', linthresh=0.01)

        # Plot for Agent 2
        ax = axes[1]
        for i, loss_type in enumerate(sorted(loss_types_agent2)):
            color = colors[i % len(colors)]
            ax.plot(self.losses_agent2[loss_type], color=color, linewidth=2, label=loss_type)
            # Add moving average for total_loss if available
            if loss_type == 'total_loss':
                window_size = min(15, len(self.losses_agent2['total_loss']))
                if window_size > 1:
                    moving_avg = np.convolve(self.losses_agent2['total_loss'],
                                            np.ones(window_size)/window_size, mode='valid')
                    ax.plot(np.arange(window_size-1, len(self.losses_agent2['total_loss'])),
                            moving_avg, 'k-', linewidth=3, alpha=0.7,
                            label=f'total_loss (MA{window_size})')
        ax.set_title(f'All Loss Components - {agent_titles[1]}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=14)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if len(loss_types_agent2) > 4:
            ncol = 2
            loc = 'upper center'
            bbox_to_anchor = (0.5, -0.1)
        else:
            ncol = 1
            loc = 'best'
            bbox_to_anchor = None
        ax.legend(loc=loc, fontsize=12, framealpha=0.7, fancybox=True, shadow=True, ncol=ncol, bbox_to_anchor=bbox_to_anchor)
        ax.set_yscale('symlog', linthresh=0.01)

        plt.suptitle('Loss Curves for Both Agents', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        plt.savefig(os.path.join(self.save_dir_png, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # No need for the separate combined total loss plot anymore since everything is combined
        # Delete the old combined_total_loss.png if it exists
        combined_loss_path = os.path.join(self.save_dir_png, 'combined_total_loss.png')
        if os.path.exists(combined_loss_path):
            try:
                os.remove(combined_loss_path)
            except:
                pass
        
    def plot_reward_histogram(self, bins=20):
        """Plot histogram of rewards to analyze distribution."""
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