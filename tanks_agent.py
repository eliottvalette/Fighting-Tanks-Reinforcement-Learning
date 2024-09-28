# tanks_agent.py
import numpy as np
from collections import deque
import random as rd
import torch
import torch.nn as nn
import torch.optim as optim
from tanks_model import TanksModel

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'

class TanksAgent(nn.Module):
    def __init__(self, state_size, action_sizes, gamma, learning_rate, load_model = False):
        super(TanksAgent, self).__init__()
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.memory = deque(maxlen=10_000)
        self.batch_size = 256

        self.model = self.build_model().to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.load_model = load_model

    def build_model(self):
        model = TanksModel(self.state_size, self.action_sizes)
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        return model

    def forward(self, state):
        return self.model(state)

    def get_actions(self, state, epsilon):
        if rd.random() <= epsilon:
            actions = [rd.randint(0, action_size - 1) for action_size in self.action_sizes]
            return actions  # Random actions for exploration

        state = torch.FloatTensor(state).to(device).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():  # No gradient computation needed for action selection
            q_values_movement, q_values_rotation, q_values_strafe, q_values_fire = self.forward(state)

        q_values = [q_values_movement, q_values_rotation, q_values_strafe, q_values_fire]

        actions = []
        for i in range(len(self.action_sizes)):
            q_values_for_action = q_values[i]  # Remove batch dimension
            best_action = torch.argmax(q_values_for_action).item()
            actions.append(best_action)

        return actions  # [int, int, int, int]

    
    def train_model(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        action = torch.LongTensor(action).to(device)
        done = torch.FloatTensor([done]).to(device)

        q_values = self.forward(state)

        with torch.no_grad(): 
            next_q_values = self.forward(next_state)

        total_loss = 0

        # Compute loss for each action dimension separately
        for i in range(len(self.action_sizes)):
            # Get current Q-value for the taken action
            current_q_value = q_values[i][action[i]]

            # Get max Q-value for next state
            max_next_q_value = torch.max(next_q_values[i])

            # Compute target Q-value
            target_q_value = reward + (1 - done) * self.gamma * max_next_q_value

            # Compute loss
            loss = self.loss_fn(current_q_value, target_q_value)
            total_loss += loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        minibatch = rd.sample(self.memory, self.batch_size)

        # Prepare batched data
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Predict Q-values for current states and next states
        q_values = self.forward(states)
        with torch.no_grad():
            next_q_values = self.forward(next_states)

        total_loss = 0

        # Compute loss for each action dimension separately
        for i in range(len(self.action_sizes)):
            # Current Q-values for the taken actions
            current_q = q_values[i].gather(1, actions[:, i].unsqueeze(1)).squeeze(1)

            # Max Q-values for the next states
            max_next_q_value = torch.max(next_q_values[i], dim=1)[0]

            # Compute target Q-values
            target_q_value = rewards + (1 - dones) * self.gamma * max_next_q_value

            # Compute loss
            loss = self.loss_fn(current_q, target_q_value)
            total_loss += loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

