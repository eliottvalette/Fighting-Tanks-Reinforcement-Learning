# tanks_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from tanks_model import ActorCriticModel
from collections import namedtuple, deque
import random
from tanks_paths import TANK_1_WEIGHTS, TANK_2_WEIGHTS

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'  # Uncomment to force CPU

class TanksAgent:
    def __init__(self, state_size, action_sizes, gamma, learning_rate, entropy_coeff=0.01, value_loss_coeff=0.5, load_model=False):
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.is_agent_1 = True  # Default to agent 1, can be changed after initialization

        self.model = ActorCriticModel(state_size, action_sizes).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)  # Experience replay buffer

        self.load_model = load_model
        if self.load_model:
            weights_file = TANK_1_WEIGHTS if self.is_agent_1 else TANK_2_WEIGHTS
            self.load(weights_file)
    
    def load(self, filename='model_weights.pth'):
        '''
        Load the model weights from the file.
        '''
        try:
            self.model.load_state_dict(torch.load(filename, map_location=device))
            print(f"Model weights loaded successfully from {filename}")
        except FileNotFoundError:
            print(f"No model weights found at {filename}. Starting with a new model.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    
    def save(self, filename='model_weights.pth'):
        '''
        Save the model weights to a file.
        '''
        try:
            torch.save(self.model.state_dict(), filename)
            print(f"Model weights saved to {filename}")
        except Exception as e:
            print(f"Error saving model weights: {e}")
    

    def get_action(self, state, epsilon, action_sizes, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor)
        self.model.train()

        actions = []
        for idx, action_prob in enumerate(action_probs[0]):  # Access first batch element
            if training and random.random() < epsilon:  # Add exploration
                action = random.randint(0, action_sizes[idx] - 1)
            else:
                if action_sizes[idx] == 3:
                    action = 0 if action_prob.item() > 0.0 else 1 if action_prob.item() < -0.0 else 2
                elif action_sizes[idx] == 2:
                    action = 0 if action_prob.item() > 0.0 else 1
                    
            actions.append(action)

        return actions

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def train_model_batch(self):
        if len(self.memory) < 16:  # Minimum batch size
            return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0, "total_loss": 0}

        batch = random.sample(self.memory, 16)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Get current policy and value predictions
        action_probs, state_values = self.model(states)

        # Compute next state values
        with torch.no_grad():
            _, next_state_values = self.model(next_states)
            next_state_values = next_state_values.squeeze(-1)

        # Compute TD targets
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = td_targets - state_values.squeeze(-1)

        # Policy loss - fixed to handle the tensor shape correctly
        policy_loss = 0
        actions_tensor = torch.tensor(actions).to(device)
        
        # Handle each action type separately
        for i in range(action_probs.size(1)):  # Iterate over each action dimension
            # Convert tanh output to probabilities for each action type
            action_dim = self.action_sizes[i]  # Get number of actions for this type
            
            # For binary actions (fire/no fire)
            if action_dim == 2:
                # Convert tanh output (-1 to 1) to probabilities for binary actions
                probs_i = torch.sigmoid(action_probs[:, i]).unsqueeze(1)
                probs = torch.cat([probs_i, 1-probs_i], dim=1)
                action_i = actions_tensor[:, i]
                log_probs = torch.log(torch.gather(probs, 1, action_i.unsqueeze(1)) + 1e-10)
                policy_loss -= torch.mean(log_probs.squeeze() * advantages.detach())
                
                # Entropy for binary actions
                entropy_loss_i = -torch.mean(probs * torch.log(probs + 1e-10))
            
            # For trinary actions (move, rotate, strafe)
            elif action_dim == 3:
                # Convert tanh output to probabilities for 3 actions using thresholds
                # Create a 3-class probability distribution
                tanh_val = action_probs[:, i]
                
                # For values > 0.2, action 0 is more likely
                # For values < -0.2, action 1 is more likely
                # Otherwise, action 2 is more likely
                probs_0 = torch.clamp((tanh_val - 0.2) / 0.8, 0, 1)
                probs_1 = torch.clamp((-0.2 - tanh_val) / 0.8, 0, 1)
                probs_2 = 1.0 - probs_0 - probs_1
                
                probs = torch.stack([probs_0, probs_1, probs_2], dim=1)
                probs = torch.clamp(probs, 1e-10, 1.0)  # Ensure valid probabilities
                probs = probs / probs.sum(dim=1, keepdim=True)  # Normalize
                
                action_i = actions_tensor[:, i]
                log_probs = torch.log(torch.gather(probs, 1, action_i.unsqueeze(1)))
                policy_loss -= torch.mean(log_probs.squeeze() * advantages.detach())
                
                # Entropy for trinary actions
                entropy_loss_i = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1))

        # Value loss
        value_loss = torch.mean((state_values.squeeze(-1) - td_targets.detach()) ** 2)

        # Entropy loss (for exploration)
        entropy_loss = entropy_loss_i  # Use the last calculated entropy

        # Total loss
        total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Return loss values for visualization
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item()
        }

    def train_model_single(self, state, actions, reward, next_state, done):
        '''
        Same as train_model_batch, but for a single experience.
        '''
        # Convert inputs to tensors
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        done = torch.FloatTensor([done]).to(device)
        actions = torch.tensor([actions]).to(device)

        # Get current policy and value predictions
        action_probs, state_values = self.model(state)

        # Compute next state values
        with torch.no_grad():
            _, next_state_values = self.model(next_state)
            next_state_values = next_state_values.squeeze(-1)

        # Compute TD target and advantage
        td_target = reward + self.gamma * next_state_values * (1 - done)
        advantage = td_target - state_values.squeeze(-1)

        # Policy loss
        policy_loss = 0
        
        # Handle each action type separately
        for i in range(action_probs.size(1)):
            action_dim = self.action_sizes[i]
            
            # For binary actions (fire/no fire)
            if action_dim == 2:
                probs_i = torch.sigmoid(action_probs[:, i]).unsqueeze(1)
                probs = torch.cat([probs_i, 1-probs_i], dim=1)
                action_i = actions[:, i]
                log_probs = torch.log(torch.gather(probs, 1, action_i.unsqueeze(1)) + 1e-10)
                policy_loss -= log_probs.squeeze() * advantage.detach()
                
                # Entropy for binary actions
                entropy_loss = -torch.mean(probs * torch.log(probs + 1e-10))
            
            # For trinary actions (move, rotate, strafe)
            elif action_dim == 3:
                tanh_val = action_probs[:, i]
                
                # Convert tanh output to probabilities for 3 actions
                probs_0 = torch.clamp((tanh_val - 0.2) / 0.8, 0, 1)
                probs_1 = torch.clamp((-0.2 - tanh_val) / 0.8, 0, 1)
                probs_2 = 1.0 - probs_0 - probs_1
                
                probs = torch.stack([probs_0, probs_1, probs_2], dim=1)
                probs = torch.clamp(probs, 1e-10, 1.0)
                probs = probs / probs.sum(dim=1, keepdim=True)
                
                action_i = actions[:, i]
                log_probs = torch.log(torch.gather(probs, 1, action_i.unsqueeze(1)))
                policy_loss -= log_probs.squeeze() * advantage.detach()
                
                # Entropy for trinary actions
                entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1))

        # Value loss
        value_loss = (state_values.squeeze(-1) - td_target.detach()) ** 2

        # Total loss
        total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Return loss values for visualization
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item()
        }
        
    def adjust_learning_rate(self, factor):
        '''
        Adjust the learning rate by multiplying it by the given factor.
        '''
        self.learning_rate *= factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate