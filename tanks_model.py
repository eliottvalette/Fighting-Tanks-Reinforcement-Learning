# tanks_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd

class ActorModel(nn.Module):
    """
    Compute the policy π_θ(a | s).

    The network builds a shared feature vector h = shared_layers(state),
    then outputs logits for all 54 possible action combinations.
    Each combination represents (move, rotate, strafe, fire) encoded as:
    idx = (((move*3 + rot)*3 + strafe)*2 + fire)

    The output is a categorical distribution over all possible action combinations.
    """
    def __init__(self, state_size, action_sizes):
        super(ActorModel, self).__init__()
        
        self.action_sizes = action_sizes
        self.n_combinations = action_sizes[0] * action_sizes[1] * action_sizes[2]  # 3*3*2 = 18

        # Enhanced shared layers with dropout for regularization
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128,32),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        # Single head for all action combinations
        self.action_head = nn.Linear(32, self.n_combinations)

    def forward(self, state):
        shared_features = self.shared_layers(state)
        
        # Output logits for all action combinations
        action_logits = self.action_head(shared_features)
        
        # Convert logits to probabilities using softmax
        action_probs = F.softmax(action_logits, dim=1)
        
        if rd.random() < 0.001:
            print('________________________')
            print(f'action_logits, mean: {action_logits.mean()}, max: {action_logits.max()}, min: {action_logits.min()}')
            print(f'action_probs, mean: {action_probs.mean()}, max: {action_probs.max()}, min: {action_probs.min()}')
            print('________________________')

        return action_probs
    
    def decode_action(self, action_idx):
        """
        Convert combined action index back to individual actions
        action_idx: Integer representing combined action
        Returns: [move, rotate, fire] actions
        """
        fire = action_idx % self.action_sizes[2]
        temp = action_idx // self.action_sizes[2]
        rotate = temp % self.action_sizes[1]
        move = temp // self.action_sizes[1]
        return [move, rotate, fire]

class CriticModel(nn.Module):
    """
    Dueling Q-network for composite actions:
        • branche partagée  → h
        • tête V(s)         → (batch,1)
        • tête A(s,a)       → (batch, 18) - one for each action combination
        • Q(s,a)=V+A-mean(A)
        
    Action combinations: 3×3×2 = 18 total combinations
    Encoded as: idx = ((move*3 + rot)*2 + fire)
    """
    def __init__(self, state_size, action_sizes):
        super().__init__()
        self.n_combinations = action_sizes[0] * action_sizes[1] * action_sizes[2]  # 3*3*2 = 18
        self.action_sizes = action_sizes
        
        self.shared = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.V_head = nn.Linear(128, 1)
        self.A_head = nn.Linear(128, self.n_combinations)

    def forward(self, state):
        """
        Here V(s) estimates the value of the state, it's an estimation of how much the situation is favorable (in terms of future expected rewards)
        A(s,a) estimates the advantage of each action combination, it's an estimation of how much each action combination is favorable compared to the other combinations.

        So Q(s,a) is a function that estimates the future expected rewards of action combination a in state s.
        To do so, it takes that current value of the state V(s), add the advantage of the action combination A(s,a) to it and substract the mean of the advantages to normalize it.
        
        In the Code, V is a tensor of shape (batch_size, 1) and A is a tensor of shape (batch_size, 54) => (batch_size, A(s, combo_0), A(s, combo_1), ..., A(s, combo_53)) 
        Q is a tensor of shape (batch_size, 54) => (batch_size, Q(s, combo_0), Q(s, combo_1), ..., Q(s, combo_53)) 
        
        Action combinations are encoded as: idx = (((move*3 + rot)*3 + strafe)*2 + fire)
        """
        h = self.shared(state)
        V = self.V_head(h)      
        A = self.A_head(h)                          
        Q = V + A - A.mean(dim=1, keepdim=True)    
        return Q, V.squeeze(-1)
    
    def get_action_combination_index(self, actions_tensor):
        """
        Convert action tensor to combination index
        actions_tensor: (batch_size, 4) where each row is [move, rotate, strafe, fire]
        Returns: (batch_size,) indices for the action combinations
        """
        # Encode as: idx = ((move*3 + rot)*2 + fire)
        combo_idx = ((actions_tensor[:, 0] * self.action_sizes[1] + actions_tensor[:, 1]) * 
                      self.action_sizes[2] + actions_tensor[:, 2])
        return combo_idx
