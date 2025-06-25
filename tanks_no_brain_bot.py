import torch
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

class NoBrainBot():
    def __init__(self, state_size, action_sizes):
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.critic = DummyCritic()
    
    def get_action(self, state, epsilon = 0.0, action_sizes = None):
        ''' 
        This function is used to determine wich action the bot will take.
        Based on the state of the game that is structured as follows:
        state = np.concatenate([
            [x_tank_1, y_tank_1],
            [direction_tank_1],
            [health_tank_1],
            [x_tank_2 - x_tank_1, y_tank_2 - y_tank_1],
            [direction_tank_2],
            [health_tank_2],
            [x_block - x_tank_1, y_block - y_tank_1],
            [x_block - x_tank_2, y_block - y_tank_2],
            [ammo_tank_1],
            [ammo_tank_2],
            [laser_distances_tank_1],
            [laser_distances_tank_2],
            [close_left],
            [close_right],
            [is_reloaded],
            [laser_distances_tank_1],
            [laser_distances_tank_2],
            [close_left],
            [close_right],
            [is_reloaded],
            [head_on_wall],
            [looking_block],
        ])

        4 actions will be taken:
        - move_action: 3 values (0, 1, 2)
        - rotate_action: 3 values (0, 1, 2)
        - strafe_action: 3 values (0, 1, 2)
        - fire_action: 2 values (0, 1)    
        '''

        # Variables from state :
        relative_direction = state[8] # range [-1, 1]
        in_sight = state[19] # range [0, 1]
        relative_position = state[4:6]
        is_reloaded = state[21]
        head_on_wall = state[22]
        looking_block = state[23]

        # Calculate distance to opponent
        distance_to_opponent = (relative_position[0]**2 + relative_position[1]**2)**0.5

        # Move action: if too far go forward (0), too close go backward (1), good distance don't move (2)
        if distance_to_opponent > 0.5:
            move_action = 0  # Forward
        elif distance_to_opponent < 0.2:
            move_action = 1  # Backward
        else:
            move_action = 2  # Don't move

        # Rotation action: turn to face opponent
        if relative_direction > 0.01:
            rotate_action = 0  # Turn right
        elif relative_direction < -0.01:
            rotate_action = 1  # Turn left
        else:
            rotate_action = 2  # Don't rotate

        # Strafe action: avoid walls and obstacles
        if head_on_wall or looking_block:
            strafe_action = 1  # Strafe right to avoid
            rotate_action = 0  # Turn right
            move_action = 0    # Forward
        else:
            strafe_action = 2  # Don't strafe

        # Fire action: if opponent in sight and we have ammo, fire
        fire_action = 1

        return [move_action, rotate_action, strafe_action, fire_action]
    
    def remember_short(self, state, actions, reward, next_state, done):
        pass
    
    def remember_long(self, state, actions, reward, next_state, done):
        pass

    def train_model_batch(self, batch_size=0, short_memory=True):
        """
        Dummy implementation to maintain compatibility with TanksAgent API.
        Since NoBrainBot is rule-based, no actual training happens.
        """
        return {"total_loss": 0, "actor_loss": 0, "critic_loss": 0}
    
    def adjust_learning_rate(self, factor):
        """
        Dummy implementation to maintain compatibility with TanksAgent API.
        """
        pass

    def save(self, filename='model_weights.pth'):
        pass
    
    def load(self, filename='model_weights.pth'):
        pass

class DummyCritic():
    def __init__(self):
        self.name = 'DummyCritic'
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.device = 'cpu'  # Force CPU for consistency
    
    def __call__(self, state):
        # Return a tuple of tensors similar to the real critic
        # First tensor: q_values (we return a dummy tensor of zeros)
        # Second tensor: value (we return a tensor with single value 0.0)
        batch_size = state.shape[0] if isinstance(state, torch.Tensor) else 1
        q_values = torch.zeros((batch_size, 54), device=self.device)  # 54 is the number from TanksAgent's critic
        value = torch.zeros((batch_size, 1), device=self.device)
        return q_values, value.squeeze(1)  # Match the expected format
    
    def eval(self):
        # No-op but needed for compatibility
        pass
    
    def train(self):
        # No-op but needed for compatibility
        pass