SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

class NoBrainBot():
    def __init__(self, state_size, action_sizes):
        self.state_size = state_size
        self.action_sizes = action_sizes
    
    def get_action(self, state):
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
        in_sight = state[18] # range [0, 1]
        relative_position = state[4:6]
        is_reloaded = state[20]
        head_on_wall = state[21]
        looking_block = state[22]

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
        if relative_direction > 0.05:
            rotate_action = 0  # Turn left
        elif relative_direction < -0.05:
            rotate_action = 1  # Turn right
        else:
            rotate_action = 2  # Don't rotate

        # Strafe action: avoid walls and obstacles
        if head_on_wall or looking_block:
            strafe_action = 0  # Strafe left to avoid
        else:
            strafe_action = 2  # Don't strafe

        # Fire action: if opponent in sight and we have ammo, fire
        if in_sight:
            fire_action = 0  # Fire
        else:
            fire_action = 1  # Don't fire

        return [move_action, rotate_action, strafe_action, fire_action]

       