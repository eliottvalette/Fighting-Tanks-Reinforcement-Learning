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
    def __init__(self, state_size, action_sizes, gamma, learning_rate, short_memory_size, long_memory_size, entropy_coeff=0.01, value_loss_coeff=0.5, load_model=False):
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.is_agent_1 = True  # Default to agent 1, can be changed after initialization

        self.model = ActorCriticModel(state_size, action_sizes).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.short_memory = deque(maxlen=short_memory_size)
        self.long_memory = deque(maxlen=long_memory_size)

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
        offset = 0
        
        # Process each action type with its corresponding probabilities
        for i, size in enumerate(action_sizes):
            # Extract the probabilities for this action type
            probs = action_probs[0, offset:offset+size]
            
            # Determine action based on probabilities or random choice for exploration
            if training and random.random() < epsilon:
                action = random.randint(0, size-1)
            else:
                # Choose action with highest probability
                action = torch.argmax(probs).item()
                
            actions.append(action)
            offset += size
            
        return actions

    def remember_short(self, state, actions, reward, next_state, done):
        self.short_memory.append((state, actions, reward, next_state, done))
    
    def remember_long(self, state, actions, reward, next_state, done):
        self.long_memory.append((state, actions, reward, next_state, done))

    def train_model_batch(self, batch_size, short_memory = False):

        if short_memory:
            memory = self.short_memory
        else:
            memory = self.long_memory

        if len(memory) < batch_size:  # Use provided batch size
            return {"policy_loss": 0, "value_loss": 0, "entropy_loss": 0, "total_loss": 0}

        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        actions_tensor = torch.tensor(actions).to(device)

        # Get current policy and value predictions
        action_probs, state_values = self.model(states)

        # Compute next state values
        with torch.no_grad():
            _, next_state_values = self.model(next_states)
            next_state_values = next_state_values.squeeze(-1)

        # Compute TD targets with clipping to prevent saturation
        next_value_term = self.gamma * next_state_values * (1 - dones)
        td_targets = rewards + next_value_term
        advantages = td_targets - state_values.squeeze(-1)

        if random.random() < 0.001:
            print(f'state_values, mean : {state_values.mean()}, max : {state_values.max()}, min : {state_values.min()}')
            print(f'rewards, mean : {rewards.mean()}, max : {rewards.max()}, min : {rewards.min()}')
            print(f'next_state_values, mean : {next_state_values.mean()}, max : {next_state_values.max()}, min : {next_state_values.min()}')
            print(f'advantages, mean : {advantages.mean()}, max : {advantages.max()}, min : {advantages.min()}')

        # Policy loss - handle each action type separately
        policy_loss = 0
        entropy_loss = 0
        offset = 0
        
        for i, size in enumerate(self.action_sizes):
            # Extract probabilities for this action type
            probs = action_probs[:, offset:offset+size]
            
            # Get the actions taken for this type
            action_i = actions_tensor[:, i]
            
            # Calculate log probabilities of chosen actions
            log_probs = torch.log(torch.gather(probs, 1, action_i.unsqueeze(1)) + 1e-10)
            
            # Policy gradient loss: -log(π(a|s)) * advantage
            policy_loss -= torch.mean(log_probs.squeeze() * advantages.detach())
            
            # Entropy loss for exploration: -Σ π(a|s) * log(π(a|s))
            entropy_i = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1))
            entropy_loss += entropy_i
            
            offset += size

        # Value loss with clipping to prevent explosion
        td_error = td_targets.detach() - state_values.squeeze(-1)
        if random.random() < 0.001:
            print(f'td_error, mean : {td_error.mean()}, max : {td_error.max()}, min : {td_error.min()}')
        # Use Smooth L1 Loss instead of MSE to be less sensitive to outliers
        value_loss = nn.SmoothL1Loss()(state_values.squeeze(-1), td_targets.detach())

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