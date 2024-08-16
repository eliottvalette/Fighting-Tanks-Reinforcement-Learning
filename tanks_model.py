# tanks_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TanksModel(nn.Module):
    def __init__(self, state_size, action_sizes):
        super(TanksModel, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.fcn1 = self.fcn2 = self.fcn3 = self.fcn4 = nn.Linear(state_size, 32)
        
        # Separate output layers for each action space
        self.movement_layer = nn.Linear(32, action_sizes[0])
        self.rotation_layer = nn.Linear(32, action_sizes[1])
        self.strafe_layer = nn.Linear(32, action_sizes[2])
        self.fire_layer = nn.Linear(32, action_sizes[3])
    
    def forward(self, state):
        x_1 = self.fcn1(state)
        x_2 = self.fcn2(state)
        x_3 = self.fcn3(state)
        x_4 = self.fcn4(state)
        q_values_movement = self.movement_layer(x_1)
        q_values_rotation = self.rotation_layer(x_2)
        q_values_strafe = self.strafe_layer(x_3)
        q_values_fire = self.fire_layer(x_4)
        
        return q_values_movement, q_values_rotation, q_values_strafe, q_values_fire