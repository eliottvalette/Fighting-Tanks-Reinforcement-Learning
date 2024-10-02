# tanks_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TanksModel(nn.Module):
    def __init__(self, state_size, action_sizes):
        super(TanksModel, self).__init__()

        # Separate networks for each action
        self.movement_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_sizes[0])
        )
        
        self.rotation_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_sizes[1])
        )
        
        self.strafe_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_sizes[2])
        )
        
        self.fire_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_sizes[3])
        )

        self.i = 0

    def forward(self, state):
        # Independent networks for each action
        movement_action = self.movement_net(state)
        rotation_action = self.rotation_net(state)
        strafe_action = self.strafe_net(state)
        fire_action = self.fire_net(state)
        
        return movement_action, rotation_action, strafe_action, fire_action
