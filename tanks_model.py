# tanks_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd

class ActorModel(nn.Module):
    """
    Compute the policy π_θ(a | s).

    The network first builds a shared feature vector h = shared_layers(state),
    then passes h through four separate heads.  Each head outputs logits that are
    converted to categorical probabilities with a softmax.  The overall policy
    for the composite action a = (a_move, a_rot, a_strafe, a_fire)  = ((a_move_1, a_move_2, a_move_3), (a_rot_1, a_rot_2, a_rot_3), (a_strafe_1, a_strafe_2, a_strafe_3), (a_fire_1, a_fire_2))
    is the product of these four independent categorical distributions.

    No extra normalisation step is required because each head already sums to 1.
    """
    def __init__(self, state_size, action_sizes):
        super(ActorModel, self).__init__()

        # Enhanced shared layers with dropout for regularization
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # Advantage stream - split into separate heads for each action type
        self.movement_stream = nn.Sequential(
            nn.Linear(128, 52),
            nn.BatchNorm1d(52),
            nn.GELU(),
            nn.Linear(52, action_sizes[0])
        )

        self.rotate_stream = nn.Sequential(
            nn.Linear(128, 52),
            nn.BatchNorm1d(52),
            nn.GELU(),
            nn.Linear(52, action_sizes[1])
        )

        self.strafe_stream = nn.Sequential(
            nn.Linear(128, 52),
            nn.BatchNorm1d(52),
            nn.GELU(),
            nn.Linear(52, action_sizes[2])
        )

        self.fire_stream = nn.Sequential(
            nn.Linear(128, 52),
            nn.BatchNorm1d(52),
            nn.GELU(),
            nn.Linear(52, action_sizes[3])
        )

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
        
        if rd.random() < 0.001:
            print('________________________')
            print(f'movement_logits, mean : {movement_logits.mean()}, max : {movement_logits.max()}, min : {movement_logits.min()}')
            print(f'action_probs_movement : {movement_probs.mean()}, max : {movement_probs.max()}, min : {movement_probs.min()}')
            print(f'rotate_logits, mean : {rotate_logits.mean()}, max : {rotate_logits.max()}, min : {rotate_logits.min()}')
            print(f'action_probs_rotate : {rotate_probs.mean()}, max : {rotate_probs.max()}, min : {rotate_probs.min()}')
            print(f'strafe_logits, mean : {strafe_logits.mean()}, max : {strafe_logits.max()}, min : {strafe_logits.min()}')
            print(f'action_probs_strafe : {strafe_probs.mean()}, max : {strafe_probs.max()}, min : {strafe_probs.min()}')
            print(f'fire_logits, mean : {fire_logits.mean()}, max : {fire_logits.max()}, min : {fire_logits.min()}')
            print(f'action_probs_fire : {fire_probs.mean()}, max : {fire_probs.max()}, min : {fire_probs.min()}')
            print('________________________')

        return action_probs

class CriticModel(nn.Module):
    """
    Dueling Q-network for composite actions:
        • branche partagée  → h
        • tête V(s)         → (batch,1)
        • tête A(s,a)       → (batch, 54) - one for each action combination
        • Q(s,a)=V+A-mean(A)
        
    Action combinations: 3×3×3×2 = 54 total combinations
    Encoded as: idx = (((move*3 + rot)*3 + strafe)*2 + fire)
    """
    def __init__(self, state_size, action_sizes):
        super().__init__()
        self.n_combinations = action_sizes[0] * action_sizes[1] * action_sizes[2] * action_sizes[3]  # 3*3*3*2 = 54
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
        # Encode as: idx = (((move*3 + rot)*3 + strafe)*2 + fire)
        combo_idx = (((actions_tensor[:, 0] * self.action_sizes[1] + actions_tensor[:, 1]) * 
                      self.action_sizes[2] + actions_tensor[:, 2]) * 
                     self.action_sizes[3] + actions_tensor[:, 3])
        return combo_idx
