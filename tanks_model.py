# tanks_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TanksModel(nn.Module):
    def __init__(self, state_size, action_sizes):
        super(TanksModel, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_size, 128), 
            nn.Linear(128, 64),         
            nn.Linear(64, 32),           
        )
        
        self.movement_layer = nn.Sequential(
            nn.Linear(32, action_sizes[0])
        )
        
        self.rotation_layer = nn.Sequential(
            nn.Linear(32, action_sizes[1])
        )
        
        self.strafe_layer = nn.Sequential(
            nn.Linear(32, action_sizes[2])
        )
        
        self.fire_layer = nn.Sequential(
            nn.Linear(32, action_sizes[3])
        )
    
    def forward(self, state):
        x = self.shared_net(state)

        # print(f"Shared net output min: {x.min():.2f}, max: {x.max():.2f}")

        q_values_movement = self.movement_layer(x)
        q_values_rotation = self.rotation_layer(x)
        q_values_strafe = self.strafe_layer(x)
        q_values_fire = self.fire_layer(x)
        return q_values_movement, q_values_rotation, q_values_strafe, q_values_fire