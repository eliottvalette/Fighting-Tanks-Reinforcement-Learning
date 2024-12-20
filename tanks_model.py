# tanks_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TanksModel(nn.Module):
    def __init__(self, state_size, action_sizes):
        super(TanksModel, self).__init__()

        self.move_size = action_sizes[0]
        self.rotate_size = action_sizes[1]
        self.strafe_size = action_sizes[2]
        self.fire_size = action_sizes[3]

        # Separate networks for each action
        self.shared_net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, action_sizes[0] + action_sizes[1] + action_sizes[2]+ action_sizes[3]),
        )

        self.softmax = nn.Softmax(dim=1)

        self.i = 0

    def forward(self, state):
        # Shared layer for all actions
        shared_actions_list = self.shared_net(state)
        movement_action = self.softmax(shared_actions_list[:, :self.move_size])
        rotation_action = self.softmax(shared_actions_list[:, self.move_size : self.move_size + self.rotate_size])
        strafe_action = self.softmax(shared_actions_list[:, self.move_size + self.rotate_size : self.move_size + self.rotate_size + self.strafe_size])
        fire_action = self.softmax(shared_actions_list[:, self.move_size + self.rotate_size + self.strafe_size:])
        
        return movement_action, rotation_action, strafe_action, fire_action
