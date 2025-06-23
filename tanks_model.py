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
            nn.Linear(state_size, 52),
            nn.BatchNorm1d(52),
            nn.GELU()
        )

        # Advantage stream - split into separate heads for each action type
        self.movement_stream = nn.Sequential(
            nn.Linear(52, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, action_sizes[0])
        )

        self.rotate_stream = nn.Sequential(
            nn.Linear(52, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, action_sizes[1])
        )

        self.strafe_stream = nn.Sequential(
            nn.Linear(52, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, action_sizes[2])
        )

        self.fire_stream = nn.Sequential(
            nn.Linear(52, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, action_sizes[3])
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
    Dueling Q-network :
        • branche partagée  → h
        • tête V(s)         → (batch,1)
        • tête A(s,a)       → (batch, N_A)
        • Q(s,a)=V+A-mean(A)
    """
    def __init__(self, state_size, action_sizes):
        super().__init__()
        self.n_actions = sum(action_sizes)           # 3+3+3+2 = 11
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.V_head = nn.Linear(128, 1)
        self.A_head = nn.Linear(128, self.n_actions)

    def forward(self, state):
        """
        Here V(s) estimates the value of the state, it's an estimation of how much the situation is favorable (in terms of future expected rewards)
        A(s,a) estimates the advantage of each action, it's an estimation of how much each action is favorable compared to the other actions.

        So Q(s,a) is a function that estimates the future expected rewards of action a in state s.
        To do so, it takes that current value of the state V(s), add the advantage of the action A(s,a) to it and substract the mean of the advantages to normalize it.
        Indeed, V(s) is a functions that already estimates the future expected rewards of the state, so we must forget to add only the advantage of an action COMPARED to the other actions.

        In the Code, V is a tensor of shape (batch_size, 1) and A is a tensor of shape (batch_size, n_actions) => (batch_size, A(s, a_1), A(s, a_2), ..., A(s, a_11)) 
        Q is a tensor of shape (batch_size, n_actions) => (batch_size, Q(s, a_1), Q(s, a_2), ..., Q(s, a_11)) 
        
        """
        h = self.shared(state)
        V = self.V_head(h)                          
        A = self.A_head(h)                          
        Q = V + A - A.mean(dim=1, keepdim=True)    
        return Q, V.squeeze(-1)                   
