# tanks_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from tanks_model import ActorCriticModel
from collections import namedtuple, deque
import random

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

        self.model = ActorCriticModel(state_size, action_sizes).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)  # Experience replay buffer

        self.load_model = load_model
        if self.load_model:
            self.load()

    def load(self):
        # Implement loading logic
        pass

    def act(self, state, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            action_probs_grouped, _ = self.model(state_tensor)
        self.model.train()

        actions = []
        for action_probs in action_probs_grouped:
            if training and random.random() < 0.1:  # Add exploration
                action = random.randint(0, action_probs.size(1) - 1)
            else:
                action = torch.argmax(action_probs, dim=1).item()
            actions.append(action)

        return actions

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < 128:  # Minimum batch size
            return

        batch = random.sample(self.memory, 128)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Get current policy and value predictions
        action_probs_grouped, state_values = self.model(states)

        # Compute next state values
        with torch.no_grad():
            _, next_state_values = self.model(next_states)
            next_state_values = next_state_values.squeeze(-1)

        # Compute TD targets
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = td_targets - state_values.squeeze(-1)

        # Policy loss
        policy_loss = 0
        for i, action_probs in enumerate(action_probs_grouped):
            action_probs = action_probs[range(len(actions)), [a[i] for a in actions]]
            policy_loss -= torch.mean(torch.log(action_probs + 1e-10) * advantages.detach())

        # Value loss
        value_loss = torch.mean((state_values.squeeze(-1) - td_targets.detach()) ** 2)

        # Entropy loss (for exploration)
        entropy_loss = 0
        for action_probs in action_probs_grouped:
            entropy_loss -= torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1))

        # Total loss
        total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
