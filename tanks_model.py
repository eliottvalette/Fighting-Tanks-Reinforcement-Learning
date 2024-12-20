# tanks_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TanksModel(nn.Module):
    def __init__(self, state_size, action_sizes):
        super(TanksModel, self).__init__()

        # Separate networks for each action
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, sum(action_sizes))
        self.leaky_relu = nn.LeakyReLU()

        self.i = 0

    def forward(self, state):
        
        # Shared layer for all actions
        x = self.fc1(state)
        x = (x - x.mean()) / (x.std() + 1e-5)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = (x - x.mean()) / (x.std() + 1e-5)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = (x - x.mean()) / (x.std() + 1e-5)
        x = self.fc4(x)
        x = self.leaky_relu(x)
        x = (x - x.mean()) / (x.std() + 1e-5)
        x = self.fc5(x)
        x = self.leaky_relu(x)
        x = (x - x.mean()) / (x.std() + 1e-5)
        x = self.fc6(x)
        x = self.leaky_relu(x)
        x = (x - x.mean()) / (x.std() + 1e-5)
        shared_actions_list = self.fc7(x)

        self.i += 1
            
        return shared_actions_list
