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
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Advantage stream - split into separate heads for each action type
        self.advantage_streams = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(action_sizes))
        )

        # Value stream with deeper architecture
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Action sizes to split the actor output
        self.action_sizes = action_sizes

        # Initialize weights with improved method
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state):
        shared_features = self.shared_layers(state)

        # Actor: Predict action probabilities for all actions using separate advantage streams
        logits = self.advantage_streams(shared_features)
        action_probs = F.tanh(logits)
        if rd.random() < 0.001:
            print('logits', logits)
            print('probs', action_probs)

        # Critic: Predict state value
        state_value = self.value_stream(shared_features)

        if rd.random() < 0.001:
            print('state_value', state_value)

        return action_probs, state_value
