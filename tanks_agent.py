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
    """
    High-level wrapper that couples an **Actor** (policy π_θ) and a **Dueling Critic**
    (Q_ϕ & V_ϕ) to control one tank.
    Key features
    ------------
    • *Composite actions* – the environment expects four independent categorical
      decisions: move (3), rotate (3), strafe (3), fire (2) ⇒ 11 discrete choices.  
    • *Two replay buffers*  
        short_memory : on-policy, 1-episode window (A2C style)  
        long_memory  : off-policy, large FIFO (DQN style)  
    • *Training loop*  
        1. Actor produces π_θ(a | s) and selects actions (ε-greedy).  
        2. Critic outputs Q(s,·) and V(s) → TD-target  
           *td* = r + γ maxₐ′ Q(s′, a′).  
        3. Losses  
           – **Actor** : −log π_θ · Advantage  (A = Q−V)  − β H[π]  
           – **Critic**: MSE(Q(s,a), td)  
        4. Two independent Adam optimisers update θ and ϕ.
    """
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
        
        # Create target critic for stable TD learning
        self.target_critic = CriticModel(state_size, action_sizes).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()  # Target network is always in eval mode
        
        # Polyak averaging parameter for target network updates
        self.tau = 0.005
        
        # Create separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        # Use lower learning rate for critic to prevent divergence
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate * 0.1)
        
        self.short_memory = deque(maxlen=short_memory_size)
        self.long_memory = deque(maxlen=long_memory_size)

        self.load_model = load_model
        if self.load_model:
            weights_file = TANK_1_WEIGHTS if self.is_agent_1 else TANK_2_WEIGHTS
            self.load(weights_file)
    
    def update_target_network(self):
        '''
        Update target network using polyak averaging
        '''
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
    
    def load(self, filename='model_weights.pth'):
        '''
        Load the model weights from the file.
        '''
        try:
            checkpoint = torch.load(filename, map_location=device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            # Also load target critic
            self.target_critic.load_state_dict(checkpoint['critic'])
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
                # Choose action with respect to the probabilities
                action = torch.distributions.Categorical(probs).sample().item()
                
            actions.append(action)
            offset += size
            
        return actions

    def remember_short(self, state, actions, reward, next_state, done):
        self.short_memory.append((state, actions, reward, next_state, done))
    
    def remember_long(self, state, actions, reward, next_state, done):
        self.long_memory.append((state, actions, reward, next_state, done))

    def train_model_batch(self, batch_size, short_memory = True):
        """
         One optimisation step on a mini-batch.

        Workflow
        --------
            1.  Sample `batch_size` transitions from the chosen replay buffer
                (short = on-policy, long = off-policy).  
            2.  Compute
                    π_θ(a|s)                         # Actor network
                    Q_ϕ(s, ·), V_ϕ(s)                # Critic network (54 combinations)  
                    Q_target(s′, ·)                  # Target critic network for TD  
                    td_target = r + γ·maxₐ′ Q_target(s′, a′)  
                    advantage = Q(s,a) − V(s)  
            3.  Losses  
                    critic_loss = Huber(Q(s,a), td_target)  
                    actor_loss  = −E[log π(a|s) · advantage] − β entropy  
            4.  Back-propagate and update the two optimisers.
            5.  Update target network with polyak averaging.
        """
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
        q_values, state_values = self.critic(states)       # Q(s,*), V(s) => (batch_size, 54), (batch_size, 1)

        # Compute next state values using target network for stability
        with torch.no_grad():
            q_next, _ = self.target_critic(next_states)           # Q_target(s',*) => (batch_size, 54)
            next_state_values = q_next.max(dim=1).values   # max_a' Q_target(s',a') => (batch_size, 1)

        # Get Q-value for the chosen action combination (no more averaging!)
        combo_idx = self.critic.get_action_combination_index(actions_tensor)
        chosen_q = q_values.gather(1, combo_idx.unsqueeze(1)).squeeze(1)  # (batch_size,)

        # Compute TD targets and advantages
        td_targets = rewards + self.gamma * next_state_values * (1 - dones) # td_target = r + γ·maxₐ′ Q_target(s′, a′)
        advantages = chosen_q.detach() - state_values      # A = Q - V Positive means the action is better than expected.

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
            probs = action_probs[:, offset:offset+size]  # (batch_size, 3) -> [0.1, 0.2, 0.7]
            
            # Get the actions taken for this type
            action_i = actions_tensor[:, i]  # (batch_size, 1) -> [2]
            
            # Calculate log probabilities of chosen actions
            log_probs = torch.log(torch.gather(probs, 1, action_i.unsqueeze(1)) + 1e-10) # inside the log, we get the probability of the chosen action than we log it. 
            
            # Policy gradient loss: -log(π(a|s)) * advantage
            policy_loss -= torch.mean(log_probs.squeeze() * advantages.detach()) # we multiply the log of the probability of the chosen action by the advantage of the action.
            
            # --- POLICY LOSS EXPLAINED ---
            # Case 1 : Bad action with high probability
            # probs = [0.1, 0.2, 0.7], action_i = [2], log_probs = log(0.7) = 0.30
            # advantage = -0.5, loss = -0.30 * -0.5 = 0.15 --> during the backprop it will push the actor to lower the prob of that action. 

            # Case 2 : Good action with low probability
            # probs = [0.1, 0.2, 0.7], action_i = [0], log_probs = log(0.1) = -2.30
            # advantage = 0.5, loss = -2.30 * 0.5 = -1.15 --> during the backprop it will push the actor to increase the prob of that action. 

            # Case 3 : Good action with high probability
            # probs = [0.1, 0.2, 0.7], action_i = [0], log_probs = log(0.7) = 0.30
            # advantage = 0.5, loss = 0.30 * 0.5 = 0.15 --> during the backprop it will push the actor to increase the prob of that action. (but not as much as the other cases)


            # Entropy loss for exploration: -Σ π(a|s) * log(π(a|s))
            entropy_i = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1)) # we sum the log of the probabilities of all actions and we multiply it by the probability of the action. So, the entropy loss is high when the actions are not uniform.
            entropy_loss += entropy_i
            
            offset += size

        # Critic loss : Huber(Q(s,a), td_target) - more robust to outliers than MSE
        td_error = td_targets.detach() - chosen_q
        if random.random() < 0.001:
            print(f'td_error, mean : {td_error.mean()}, max : {td_error.max()}, min : {td_error.min()}')
        critic_loss = torch.nn.functional.smooth_l1_loss(chosen_q, td_targets.detach())

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
        
        # Update target network with polyak averaging
        self.update_target_network()
        
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
            param_group['lr'] = self.learning_rate * 0.1  # Keep critic LR lower