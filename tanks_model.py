# tanks_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd

class ActorCriticModel(nn.Module):
    def __init__(self, state_size, action_sizes):
        super(ActorCriticModel, self).__init__()

        # Enhanced shared layers with dropout for regularization
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 52),
            nn.BatchNorm1d(52),
            nn.GELU()

        )

        # Advantage stream - split into separate heads for each action type
        self.movement_stream = nn.Sequential(
            nn.Linear(52, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

        self.rotate_stream = nn.Sequential(
            nn.Linear(52, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

        self.strafe_stream = nn.Sequential(
            nn.Linear(52, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

        self.fire_stream = nn.Sequential(
            nn.Linear(52, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

        # Value stream with deeper architecture
        self.value_stream = nn.Sequential(
            nn.Linear(52, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # Action sizes to split the actor output
        self.action_sizes = action_sizes

    def forward(self, state):
        shared_features = self.shared_layers(state)

        # Actor: Predict action probabilities for all actions using separate advantage streams
        movement_logits = self.movement_stream(shared_features)
        rotate_logits = self.rotate_stream(shared_features)
        strafe_logits = self.strafe_stream(shared_features)
        fire_logits = self.fire_stream(shared_features)

        # Convert logits to probabilities using softmax
        movement_probs = F.softmax(movement_logits, dim=1)
        rotate_probs = F.softmax(rotate_logits, dim=1)
        strafe_probs = F.softmax(strafe_logits, dim=1)
        fire_probs = F.softmax(fire_logits, dim=1)

        action_probs = torch.cat([movement_probs, rotate_probs, strafe_probs, fire_probs], dim=1)
        
        # Critic: Predict state value with scaling to prevent large values
        state_value = self.value_stream(shared_features) * 0.1  # Increased scaling factor for better value learning
        
        if rd.random() < 0.001:
            print('________________________')
            print('movement_logits :', movement_logits)
            print('action_probs_movement :', movement_probs)
            print('rotate_logits :', rotate_logits)
            print('action_probs_rotate :', rotate_probs)
            print('strafe_logits :', strafe_logits)
            print('action_probs_strafe :', strafe_probs)
            print('fire_logits :', fire_logits)
            print('action_probs_fire :', fire_probs)
            print('state_value :', state_value)
            print('________________________')

        return action_probs, state_value
