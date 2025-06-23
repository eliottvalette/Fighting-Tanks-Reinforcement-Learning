# tanks_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from tanks_model import ActorModel, CriticModel
from collections import namedtuple, deque
import random
from tanks_paths import TANK_1_WEIGHTS, TANK_2_WEIGHTS

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'  # Uncomment to force CPU

class TanksAgent:
    def __init__(self, state_size, action_sizes, gamma, learning_rate, short_memory_size, long_memory_size, entropy_coeff=0.01, policy_loss_coeff=1.0, load_model=False):
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coeff = entropy_coeff
        self.policy_loss_coeff = policy_loss_coeff
        self.is_agent_1 = True  # Default to agent 1, can be changed after initialization

        # Create separate actor and critic models
        self.actor = ActorModel(state_size, action_sizes).to(device)
        self.critic = CriticModel(state_size, action_sizes).to(device)
        
        # Create separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
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
            checkpoint = torch.load(filename, map_location=device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
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
            checkpoint = {
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict()
            }
            torch.save(checkpoint, filename)
            print(f"Model weights saved to {filename}")
        except Exception as e:
            print(f"Error saving model weights: {e}")
    
    def get_action(self, state, epsilon, action_sizes, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        self.actor.train()

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
            return {"total_loss": 0, "actor_loss": 0, "critic_loss": 0}

        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        actions_tensor = torch.tensor(actions).to(device)

        # Get current policy and value predictions
        action_probs = self.actor(states)
        q_values, state_values = self.critic(states)       # Q(s,*), V(s)

        # Compute next state values
        with torch.no_grad():
            q_next, _ = self.critic(next_states)           # Q(s',*)
            next_state_values = q_next.max(dim=1).values   # max_a' Q(s',a')

        # Select Q(s,a) played (average over all action heads)
        chosen_q = []
        offset = 0
        for i, size in enumerate(self.action_sizes):
            a_i = actions_tensor[:, i]                     # (B,)
            q_i  = q_values[:, offset:offset+size]         # (B,size)
            chosen_q.append( q_i.gather(1, a_i.unsqueeze(1)) )
            offset += size
        chosen_q = torch.cat(chosen_q, dim=1).mean(dim=1)  # (B,)  moyenne des 4 têtes

        # Compute TD targets and advantages
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = chosen_q.detach() - state_values      # A = Q - V

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

        # Critic loss : MSE(Q(s,a), td_target)
        td_error = td_targets.detach() - chosen_q
        if random.random() < 0.001:
            print(f'td_error, mean : {td_error.mean()}, max : {td_error.max()}, min : {td_error.min()}')
        critic_loss = torch.nn.functional.mse_loss(chosen_q, td_targets.detach())

        # Backpropagation for actor (policy network)
        actor_loss = self.policy_loss_coeff * policy_loss - self.entropy_coeff * entropy_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Backpropagation for critic (value network)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Calculate total loss for logging purposes
        total_loss = actor_loss + critic_loss
        
        # Return loss values for visualization
        return {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }
        
    def adjust_learning_rate(self, factor):
        '''
        Adjust the learning rate by multiplying it by the given factor.
        '''
        self.learning_rate *= factor
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.learning_rate