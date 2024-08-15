# tanks_agent.py
import numpy as np
from collections import deque
import random as rd
import torch
import torch.nn as nn
import torch.optim as optim
from tanks_model import TanksModel

class TanksAgent(nn.Module):
    def __init__(self, state_size, action_sizes, gamma, learning_rate, load_model = False):
        super(TanksAgent, self).__init__()
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.memory = deque(maxlen=10_000)
        self.batch_size = 256

        self.model = self.build_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.load_model = load_model

    def build_model(self):
        return TanksModel(self.state_size, self.action_sizes)

    def forward(self, state):
        return self.model(state)

    def get_actions(self, state, epsilon):
        if rd.random() <= epsilon:
            actions = [rd.randint(0,action_size - 1) for action_size in self.action_sizes]
            return actions
        
        state = torch.FloatTensor(state)

        # Unpack the Q-values tuple returned by the model
        q_values_movement, q_values_rotation, q_values_strafe, q_values_fire = self.forward(state)

        q_values = [q_values_movement, q_values_rotation, q_values_strafe, q_values_fire]

        actions = []
        for i in range(len(self.action_sizes)):
            q_values_for_action = q_values[i]  # Extract Q-values for this particular action
            best_action = torch.argmax(q_values_for_action).item()
            actions.append(best_action)
        
        return actions
    
    def train_model(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        action = torch.LongTensor(action)
        done = torch.FloatTensor([done])

        q_values = self.forward(state)
        next_q_values = self.forward(next_state)

        target_q_values = reward.repeat(len(self.action_sizes))

        # Calculate current Q-value for the taken actions
        current_q_values = torch.stack([q_values[i][action[i]] for i in range(len(self.action_sizes))])

        for i in range(len(self.action_sizes)):
            max_next_q_value = torch.max(next_q_values[i])  # Max Q-value for the next state in this action space
            target_q_values += (1 - done) * self.gamma * max_next_q_value

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = rd.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.train_model(state, action, reward, next_state, done)
