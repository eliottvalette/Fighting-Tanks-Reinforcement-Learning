# tanks_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticModel(nn.Module):
    def __init__(self, state_size, action_sizes):
        super(ActorCriticModel, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Separate streams for actor and critic
        self.actor_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, sum(action_sizes)),
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Action sizes to split the actor output
        self.action_sizes = action_sizes

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(module.bias)

    def forward(self, state):
        shared_features = self.shared_layers(state)

        # Actor: Predict action probabilities for all actions
        action_logits = self.actor_layers(shared_features)
        action_logits_grouped = torch.split(action_logits, self.action_sizes, dim=1)
        action_probs_grouped = [F.softmax(logits, dim=1) for logits in action_logits_grouped]

        # Critic: Predict state value
        state_value = self.critic_layers(shared_features).squeeze(-1)

        return action_probs_grouped, state_value
