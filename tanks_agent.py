# tanks_agent.py
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tanks_model import TanksModel

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# device = 'cpu'  # Uncomment this line if you want to force CPU

print(f"Device: {device}")

class TanksAgent:
    def __init__(self, state_size, action_sizes, gamma, learning_rate, load_model=False):
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.model = self.build_model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.load_model = load_model

        if self.load_model:
            self.load()

    def build_model(self):
        model = TanksModel(self.state_size, self.action_sizes)
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        return model

    def load(self):
        # Implement model loading logic if necessary
        pass

    def get_actions(self, state, epsilon):
        """
        Selects actions for each action dimension using epsilon-greedy policy.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Shape: (1, state_size)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state)  # List of tensors
        self.model.train()

        actions = []
        q_values_grouped = torch.split(q_values, self.action_sizes, dim=1)
        for q in q_values_grouped:
            if random.random() < epsilon:
                action = random.randint(0, len(q) - 1)
            else:
                action = torch.argmax(q, dim=1).item()
            actions.append(action)
        return actions # List of ints

    def train_model(self, state, actions, reward, next_state, done):
        """
        Trains the model using the Q-learning update rule.
        """
        # Convert state and next_state to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)            # Shape: (1, state_size)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)  # Shape: (1, state_size)
        actions_tensor = torch.LongTensor(actions).to(device)                      # List of ints
        reward_tensor = torch.FloatTensor([reward]).to(device)                     # Shape: (1,)
        done_tensor = torch.FloatTensor([done]).to(device)                         # Shape: (1,)

        # Get current Q-values
        q_values = self.model(state_tensor)  # Shape: (1, total_action_sizes)
        q_values_grouped = torch.split(q_values, self.action_sizes, dim=1)  # Tuple of tensors

        # Get next Q-values
        with torch.no_grad():
            q_next = self.model(next_state_tensor)
            q_next_grouped = torch.split(q_next, self.action_sizes, dim=1)

        # Initialize list for targets and predictions
        targets = []
        predictions = []

        # Iterate over each action dimension
        for idx, (q, q_nxt) in enumerate(zip(q_values_grouped, q_next_grouped)):
            action = actions_tensor[idx]  # Action taken in this dimension

            # Predicted Q-value for the taken action
            q_pred = q[0, action]

            if done_tensor.item() == 1:
                # If done, target is just the reward
                target = reward_tensor
            else:
                # Target is reward + gamma * max Q(next_state, a')
                target = reward_tensor + self.gamma * torch.max(q_nxt, dim=1)[0]

            predictions.append(q_pred)
            targets.append(target)

        # Stack predictions and targets
        predictions = torch.stack(predictions)  # Shape: (num_action_dims,)
        targets = torch.stack(targets).squeeze()  # Shape: (num_action_dims,)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()


