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
            nn.Linear(state_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        # Advantage stream - split into separate heads for each action type
        self.movement_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

        self.rotate_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

        self.strafe_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

        self.fire_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        # Value stream with deeper architecture
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # Action sizes to split the actor output
        self.action_sizes = action_sizes

        # Initialize weights with improved method
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)  # Using smaller gain to prevent large outputs
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        # Handle single-sample case (when not training with batches)
        batch_size = state.size(0)
        was_single = False
        
        if batch_size == 1 and self.training:
            # Temporarily switch to eval mode for BatchNorm
            self.eval()
            was_single = True
            
        shared_features = self.shared_layers(state)

        # Actor: Predict action probabilities for all actions using separate advantage streams
        movement_probs = self.movement_stream(shared_features)
        rotate_probs = self.rotate_stream(shared_features)
        strafe_probs = self.strafe_stream(shared_features)
        fire_probs = self.fire_stream(shared_features)

        action_probs = torch.cat([movement_probs, rotate_probs, strafe_probs, fire_probs], dim=1)
        
        # Critic: Predict state value with scaling to prevent large values
        state_value = self.value_stream(shared_features) * 0.01  # Reduced scaling factor
        
        # Switch back to training mode if we temporarily changed it
        if was_single:
            self.train()
            
        if rd.random() < 0.001 and not was_single:
            print('________________________')
            print('action_probs_movement :', movement_probs)
            print('action_probs_rotate :', rotate_probs)
            print('action_probs_strafe :', strafe_probs)
            print('action_probs_fire :', fire_probs)
            print('state_value :', state_value)
            print('________________________')

        return action_probs, state_value
