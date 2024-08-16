# tanks_game.py
import math
import pygame
import numpy as np
import random as rd
import time

from tanks_game_objects import Background, TankPlayer, Bullet, Block
from tanks_paths import BACKGROUND, TANK_1_IMAGE, TANK_2_IMAGE, BULLET_IMAGE, CRATE_IMAGE


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
TANK_1_SPEED = 5 # Tank 1 is faster
TANK_2_SPEED = 3
ROTATION_ANGLE_1 = 1 # But rotates slower
ROTATION_ANGLE_2 = 2
TANK_SIZE = 70
BULLET_DAMAGE = 25
BLOCK_SIZE = 100
LASER_MAX_SIZE = int(np.sqrt(SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2))
background = Background(image_file = BACKGROUND, location = [0,0], width = SCREEN_WIDTH, height = SCREEN_HEIGHT)

class TanksGame:
    def __init__(self):
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen_dims = [SCREEN_WIDTH, SCREEN_HEIGHT]
        self.clock  = pygame.time.Clock()
        self.position_1 = [100, 180]
        self.tank_1 = TankPlayer(image_file = TANK_1_IMAGE, location = self.position_1, width = TANK_SIZE, speed = TANK_1_SPEED)
        self.tank_1.rotate(-30) 
        self.position_2 = [SCREEN_WIDTH - 100, SCREEN_HEIGHT - 180]
        self.tank_2 = TankPlayer(image_file = TANK_2_IMAGE, location = self.position_2, width = TANK_SIZE, speed = TANK_2_SPEED)
        self.tank_2.rotate(150)

        self.middle_block = Block(image_file = CRATE_IMAGE, location=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), width=BLOCK_SIZE, height=BLOCK_SIZE * 2)

    def reset(self):
        self.position_1 = [100, 180]
        self.tank_1 = TankPlayer(image_file = TANK_1_IMAGE, location = self.position_1, width = TANK_SIZE, speed = TANK_1_SPEED)
        self.tank_1.rotate(-30) 
        self.position_2 = [SCREEN_WIDTH - 100, SCREEN_HEIGHT - 180 ]
        self.tank_2 = TankPlayer(image_file = TANK_2_IMAGE, location = self.position_2, width = TANK_SIZE, speed = TANK_2_SPEED)
        self.tank_2.rotate(150) 


        
    def rotate_tank(self, num_tank, rotation_direction):
        if rotation_direction == 'right':
            direction_of_rotation = - 1
        else :
            direction_of_rotation = 1
        if num_tank == 1:
            self.tank_1.rotate(ROTATION_ANGLE_1 * direction_of_rotation)
        else:
            self.tank_2.rotate(ROTATION_ANGLE_2 * direction_of_rotation)

    def calculate_movement(self, num_tank, angle_offset=0, slowdown= 1):
        tank = getattr(self, f'tank_{num_tank}')
        rad_angle =  np.radians(tank.direction + angle_offset)
        dx = tank.speed * np.cos(rad_angle) * slowdown
        dy = - tank.speed * np.sin(rad_angle) * slowdown  # Negative because Pygame's y-axis is flipped
        return dx, dy

    def move(self, num_tank = 1, dx = 0, dy = 0):
        if num_tank == 1 :
            self.position_1[0] += dx
            self.position_1[1] += dy
        elif num_tank == 2 :
            self.position_2[0] += dx
            self.position_2[1] += dy 

    def strafe_left(self, num_tank):
        dx, dy = self.calculate_movement(num_tank, angle_offset=90, slowdown = 0.3) # Strafe very slowly (30% of the speed)
        self.move(num_tank, dx, dy)

    def strafe_right(self, num_tank):
        dx, dy = self.calculate_movement(num_tank, angle_offset=-90, slowdown = 0.3)
        self.move(num_tank, dx, dy)

    def move_forward(self, num_tank):
        dx, dy = self.calculate_movement(num_tank)
        self.move(num_tank, dx, dy)

    def move_backward(self, num_tank):
        dx, dy = self.calculate_movement(num_tank, slowdown = 0.6) # Go backward slowly (60% of the speed)
        self.move(num_tank, -dx, -dy)
    
    def update_tank_position(self):
        for i in [1, 2]:
            tank = getattr(self, f'tank_{i}')
            position = getattr(self, f'position_{i}')  

            position[0] = np.clip(position[0], 30, SCREEN_WIDTH - 30)# restrain position[0] between 0 and SCREEN_WIDTH
            position[1] = np.clip(position[1], 30, SCREEN_HEIGHT - 30)

            x_top_left, y_top_left = self.middle_block.rect.topleft
            x_bottom_right, y_bottom_right = self.middle_block.rect.bottomright

            # Clip position to avoid the block, clip is obviously not the best choice but I found it more visual
            if x_top_left <= position[0] <= x_bottom_right and position[1] <= y_top_left:
                position[1] = np.clip(position[1], 30, y_top_left - 30)

            if x_top_left <= position[0] <= x_bottom_right and position[1] >= y_bottom_right:
                position[1] = np.clip(position[1], y_bottom_right + 30, SCREEN_HEIGHT - 30)

            if y_top_left <= position[1] <= y_bottom_right and position[0] <= x_top_left:
                position[0] = np.clip(position[0], 30, x_top_left - 30)

            if y_top_left <= position[1] <= y_bottom_right and position[0] >= x_bottom_right:
                position[0] = np.clip(position[0], x_bottom_right + 30, SCREEN_WIDTH - 30)
            
            # Update tank position
            tank.rect.center = position[0] , position[1]


    def fire_bullet(self, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position = getattr(self, f'position_{num_tank}')
        x, y = position

        direction = tank.direction
        current_time = time.time()

        if tank.health > 0 and tank.number_of_ammo > 0 and tank.check_cooldown(current_time):
            # Fire the bullet and reset cooldown
            bullet = Bullet(image_file=BULLET_IMAGE, location=(x, y), 
                            direction=direction, width=10, screen_dims=(SCREEN_WIDTH, SCREEN_HEIGHT),
                            block=self.middle_block)

            tank.number_of_ammo -= 1
            tank.bullets.add(bullet)
            tank.cooldown = current_time  # Reset the cooldown only if the bullet is fired
    
    def check_bullet_collisions(self): # Collisions with tanks and blocks
        tanks = [self.tank_1, self.tank_2]
        for i, tank in enumerate(tanks):
            for bullet in tank.bullets:
                opponent_tank = self.tank_1 if i == 1 else self.tank_2
                if bullet.check_collision(opponent_tank.rect):
                    self.handle_collision(bullet, opponent_tank, 2 - i)
                elif bullet.block and bullet.rect.colliderect(bullet.block.rect):
                    self.handle_block_collision(bullet)
    
    def update_bullets(self): # In case of going OOB
        for tank in [self.tank_1, self.tank_2]:
            for bullet in tank.bullets:
                bullet.update()
                if bullet.rect.x < 0 or bullet.rect.x > self.screen_dims[0] or bullet.rect.y < 0 or bullet.rect.y > self.screen_dims[1]:
                    tank.bullets.remove(bullet)


    def handle_block_collision(self, bullet): # Remove the bullet from the sprite.Group
        bullet.kill()
        
    def handle_collision(self, bullet, tank, num_tank):
        tank.health -= BULLET_DAMAGE
        bullet.kill()
        tank.was_hit = True
        print(f"Tank {num_tank} was hit!")
    


    def cast_laser(self, direction, max_distance, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position =  getattr(self, f'position_{num_tank}')

        opponent_tank = getattr(self, f'tank_{3-num_tank}')
        x, y = position
        angle_rad = math.radians(direction)

        for distance in range(1, max_distance + 1):
            ray_x = x + distance * math.cos(angle_rad) # this is the point, at the end of the ray [tank]-------* 
            ray_y = y + distance * math.sin(angle_rad)

            # Check if the ray hits a border
            if ray_x < 0 or ray_y < 0 or ray_x >= SCREEN_WIDTH or ray_y >= SCREEN_HEIGHT:
                return distance
            
            if self.middle_block.rect.collidepoint(ray_x, ray_y):
                return distance
            
            compensated_laser_direction= (direction + tank.direction)%360
            if compensated_laser_direction == 0 :
                tank.in_line_of_sight = False
            if compensated_laser_direction == 20 :
                tank.on_close_right = False
            if compensated_laser_direction == 340 :
                tank.on_close_left = False

            if opponent_tank.rect.collidepoint(ray_x, ray_y):
                if compensated_laser_direction == 0:
                    tank.in_line_of_sight = True
                if compensated_laser_direction == 20: # a bit to much on the right
                    tank.on_close_right = True
                if compensated_laser_direction == 340:  # a bit to much on the left
                    tank.on_close_left = True
                return distance
            
        return max_distance  # If no collision, return max_distance

    def get_all_laser_distances(self, max_distance):
        directions = [0, 20, 45, 90, 135, 180, 225, 270, 315, 340]  # 10 directions (degrees)
        
        # Adjust directions relative to tank_1's direction
        adjusted_directions_tank_1 = [(direction - self.tank_1.direction) % 360 for direction in directions] # adding tha angle of the tank modulo 360
        distances_tank_1 = [self.cast_laser(direction, max_distance, num_tank=1) for direction in adjusted_directions_tank_1]

        # Adjust directions relative to tank_2's direction
        adjusted_directions_tank_2 = [(direction - self.tank_2.direction) % 360 for direction in directions]
        distances_tank_2 = [self.cast_laser(direction, max_distance, num_tank=2) for direction in adjusted_directions_tank_2]
        
        return [distances_tank_1, distances_tank_2]

    def get_angle_to_opponent(self, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position = np.array(getattr(self, f'position_{num_tank}'))
        
        opponent_position = np.array(getattr(self, f'position_{3 - num_tank}'))

        # Calculate relative position
        delta_x = opponent_position[0] - position[0]
        delta_y = opponent_position[1] - position[1]

        # Calculate angle of the line / Arctan2(x,y) computes the angle for any point in the 2D plane. This means it can return values from -pi to pi, covering all four quadrants.
        angle_to_opponent = np.degrees(np.arctan2(delta_y, delta_x))

        # Calculate the relative angle with respect to the ally tank's direction
        relative_angle = (angle_to_opponent + tank.direction + 180) % 360 # +180 in order to normalize it

        return relative_angle

    def standardize(self, value, mean, std):
        return (value - mean) / std

    def normalize(self, value, max_value):
        return value / max_value

    def get_state(self, num_tank):
        # Retrieve tank and opponent details
        tank = getattr(self, f'tank_{num_tank}')
        opponent_tank = getattr(self, f'tank_{3 - num_tank}')

        position = np.array(getattr(self, f'position_{num_tank}'))
        opponent_position = np.array(getattr(self, f'position_{3 - num_tank}'))

        health = tank.health
        direction = tank.direction
        opponent_health = opponent_tank.health
        opponent_direction = opponent_tank.direction

        relative_angle_toward_opponent = self.get_angle_to_opponent(num_tank)
        ammo = tank.number_of_ammo
        in_sight = tank.in_line_of_sight
        close_right = tank.on_close_right
        close_left = tank.on_close_left
        is_reloaded = tank.check_cooldown(current_time=time.time())

        distance_to_opponent = np.linalg.norm(opponent_position - position)
        distance_to_block = np.linalg.norm(position - np.array(self.middle_block.rect.center))


        # Calculate relative opponent position
        relative_position = opponent_position - position  # Translate to ally tank's position
        angle_rad = np.radians(-direction)  # Convert ally tank's direction to radians (negative for rotation)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        relative_position = np.dot(rotation_matrix, relative_position)  # Rotate to ally tank's orientation

        # Get laser distances
        laser_distances = np.array(self.get_all_laser_distances(800)[num_tank - 1])

        # Define means, stds, and max values
        means = {
            "position": np.array([400, 400]),
            "direction": 180,
            "opponent_direction": 180,
            "relative_angle": 180,
        }

        stds = {
            "position": np.array([400, 400]),
            "direction": 180,
            "opponent_direction": 180,
            "relative_angle": 180,
        }

        maxs = {
            "relative_position": np.array([800, 800]),
            "health": 100,
            "opponent_health": 100,
            "ammo": 1000,
            "laser_distances": LASER_MAX_SIZE,
        }

        # Standardize / Normalize each feature
        position_standardized           = self.standardize(position, means["position"], stds["position"])
        direction_standardized          = self.standardize(direction, means["direction"], stds["direction"])
        opponent_direction_standardized = self.standardize(opponent_direction, means["opponent_direction"], stds["opponent_direction"])
        relative_angle_standardized     = self.standardize(relative_angle_toward_opponent, means["relative_angle"], stds["relative_angle"])
        
        relative_position_normalized    = self.normalize(relative_position, maxs["relative_position"])
        health_normalized          = self.normalize(health, maxs["health"])
        opponent_health_normalized = self.normalize(opponent_health, maxs["opponent_health"])
        ammo_normalized            = self.normalize(ammo, maxs["ammo"])
        laser_distances_normalized = self.normalize(laser_distances, maxs["laser_distances"])
        distance_normalized        = self.normalize(distance_to_opponent, np.linalg.norm([SCREEN_WIDTH, SCREEN_HEIGHT]))
        distance_block_normalized  = self.normalize(distance_to_block, np.linalg.norm([SCREEN_WIDTH, SCREEN_HEIGHT]))


        state = np.concatenate([
            position_standardized,              # Standardized Position (Size: 2)
            [direction_standardized],           # Standardized Direction (Size: 1)
            # [health_normalized],                # Normalized Health (Size: 1)
            relative_position_normalized,       # Normalized Relative Opponent Position (Size: 2)
            # [opponent_direction_standardized],  # Standardized Opponent Direction (Size: 1)
            # [opponent_health_normalized],       # Normalized Opponent Health (Size: 1)
            [relative_angle_standardized],      # Standardized Relative Angle to Opponent (Size: 1)
            [distance_normalized],              # Normalized Distance to opponent (Size: 1)
            [distance_block_normalized],        # Normalized Distance to block (Size: 1)
            # [ammo_normalized],                  # Normalized Ammo count (Size: 1)
            laser_distances_normalized[[0, 1, 9]],         # Normalized Laser distances (Size: 10)
            [close_left],                       # Close Left Boolean (Size: 1)
            [in_sight],                         # In Sight Boolean (Size: 1)
            [close_right],                      # Close Right Boolean (Size: 1)
            # [is_reloaded],                      # Is Reloaded Boolean (Size: 1)
        ])

        return state

    def is_head_against_the_wall(self, laser_distances):
        right = laser_distances[0] < 0.075
        middle = laser_distances[1] < 0.06
        left = laser_distances[-1] < 0.075
        return right and middle and left
    
    def step(self, actions, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position = getattr(self, f'position_{num_tank}')
        opponent_tank = getattr(self, f'tank_{3 - num_tank}')
        opponent_position = getattr(self, f'position_{3 - num_tank}')

        tank.reward=0

        move_action, rotate_action, strafe_action, fire_action = actions

        previous_distance_between_them = np.linalg.norm(np.array(position) - np.array(opponent_position))
        previous_relative_angle_toward_opponent = abs(self.get_angle_to_opponent(num_tank) / 180 - 1)
        
        if move_action == 0:
            self.move_forward(num_tank)
        elif move_action == 1:
            self.move_backward(num_tank)
        
        if rotate_action == 0:
            self.rotate_tank(num_tank, rotation_direction= 'right')
        elif rotate_action == 1:
            self.rotate_tank(num_tank, rotation_direction= 'left')
        
        if strafe_action == 0:
            self.strafe_left(num_tank)
        elif strafe_action == 1:
            self.strafe_right(num_tank)

        if fire_action == 0:
            self.fire_bullet(num_tank)

        new_distance_between = np.linalg.norm(np.array(position) - np.array(opponent_position))
        new_relative_angle_toward_opponent = abs(self.get_angle_to_opponent(num_tank) / 180 - 1)

        self.update_bullets()  # Ensure bullets are updated every step
        self.check_bullet_collisions()  # Ensure collisions are checked every step

        laser_distances = np.array(self.get_all_laser_distances(800)[num_tank - 1])/800

        if self.lost(tank):
            tank.reward -= 50  # Penalty for losing
            done = True
        elif self.lost(opponent_tank):
            tank.reward += 100  # Big reward for winning
            done = True
        else:
            done = False
            if new_distance_between <  previous_distance_between_them : # Reward if the tanks got closer
                tank.reward += 6
            else :
                tank.reward -= 3
            
            if new_relative_angle_toward_opponent < previous_relative_angle_toward_opponent : 
                tank.reward += 4
            else :
                tank.reward -= 2

            if self.is_head_against_the_wall(laser_distances) and 1 not in [tank.in_line_of_sight, tank.on_close_right, tank.on_close_left]:
                tank.reward -= 10  # Discourage bumping his head on the wall

            if opponent_tank.was_hit :
                tank.reward += 50  # Reward for hitting the opponent
                opponent_tank.was_hit = False

            if tank.was_hit :
                tank.reward -= 20  # Penalty for getting hit
                tank.was_hit = False

            if tank.in_line_of_sight:
                tank.reward += 10  # Encourage aiming at the opponent
            
            if tank.on_close_right or tank.on_close_left:
                tank.reward += 2  # Encourage aiming close to the opponent

            if fire_action == 0 and tank.in_line_of_sight:
                tank.reward += 50 # Reward fire action in sight

            if tank.number_of_ammo == 0:
                tank.reward -= 60  # Penalty for running out of ammo

        tank.total_reward += tank.reward
        return self.get_state(num_tank), tank.reward, done, {}

    def lost(self, tank):
        return tank.health <= 0



    def draw_hitboxes(self, draw = False):
        if draw :
            # Draw hitboxes around the tanks
            pygame.draw.rect(self.screen, (161, 155, 88), self.tank_1.rect, 2)  # Desert hitbox for tank 1
            pygame.draw.rect(self.screen, (41, 79, 23), self.tank_2.rect, 2)    # Green hitbox for tank 2

    def draw_laser(self, tank, position, laser_distances, num_tank):
        laser_angles = [0, 20, 45, 90, 135, 180, 225, 270, 315, 340]
        for i, direction in enumerate(laser_angles):
            direction -= tank.direction
            distance = laser_distances[i]
            angle_rad = math.radians(direction)
            start_pos = (position[0], position[1])
            end_pos = (
                position[0] + distance * math.cos(angle_rad),
                position[1] + distance * math.sin(angle_rad),
            )
            if num_tank == 1:
                color = (161, 155, 88)
            else :
                color = (41, 79, 23) 
            if (i == 0 and tank.in_line_of_sight) or (i == 1 and tank.on_close_right) or (i == 9 and tank.on_close_left):
                    color = (255, 0, 0)
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
    
    def draw_text(self, tank, num_tank, epsilon = None):
        score_font = pygame.font.Font(pygame.font.get_default_font(), 14)
        background_color = (161, 155, 88)  # White semi-transparent background

        if num_tank == 1 :
            score_text = score_font.render(f"[Tank 1] Current reward : {tank.total_reward} Health = {tank.health} %", True, (0, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.topleft = (10, 10)
        else :
            score_text = score_font.render(f"[Tank 2] Current reward : {tank.total_reward} Health = {tank.health} %", True, (0, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.topright = (SCREEN_WIDTH -10, 10)

        # Draw background rectangle behind the text
        pygame.draw.rect(self.screen, background_color, score_rect.inflate(10, 10))
        self.screen.blit(score_text, score_rect)

        if epsilon is not None :
            epsilon_text = score_font.render(f"Randomness: {epsilon*100:.1f}%", True, (0, 0, 0))
            epsilon_rect = epsilon_text.get_rect()
            epsilon_rect.topleft = (SCREEN_WIDTH // 2 - 50, 10)
            
            # Draw background rectangle behind the epsilon text
            pygame.draw.rect(self.screen, background_color, epsilon_rect.inflate(10, 10))
            self.screen.blit(epsilon_text, epsilon_rect)

    def minimal_render(self, rendering):
        if rendering :
            self.screen.fill([255, 255, 255])
            self.update_tank_position()

            self.update_bullets()
            self.check_bullet_collisions()

            self.draw_hitboxes(True)

            max_distance = LASER_MAX_SIZE
            laser_distances = self.get_all_laser_distances(max_distance)

            self.draw_laser(tank = self.tank_1, position = self.position_1, laser_distances = laser_distances[0], num_tank = 1)
            self.draw_laser(tank = self.tank_2, position = self.position_2, laser_distances = laser_distances[1], num_tank = 2)

            pygame.display.flip()

    def render(self, rendering, clock, epsilon = 0):
        if rendering :
            self.screen.fill([255, 255, 255])
            self.screen.blit(background.image, background.rect)

            # Draw the middle block
            self.screen.blit(self.middle_block.image, self.middle_block.rect)

            # Draw the tank
            self.update_tank_position()
            if self.tank_1.health  > 0 :
                self.screen.blit(self.tank_1.image, self.tank_1.rect)
            if self.tank_2.health  > 0 :
                self.screen.blit(self.tank_2.image, self.tank_2.rect)

            self.draw_hitboxes(True)

            max_distance = LASER_MAX_SIZE  # You can adjust this value
            laser_distances = self.get_all_laser_distances(max_distance)

            self.draw_laser(tank = self.tank_1, position = self.position_1, laser_distances = laser_distances[0], num_tank = 1)
            self.draw_laser(tank = self.tank_2, position = self.position_2, laser_distances = laser_distances[1], num_tank = 2)

            self.draw_text(tank = self.tank_1, num_tank = 1, epsilon = epsilon)
            self.draw_text(tank = self.tank_2, num_tank = 2)

            # Update bullets
            for tank in [self.tank_1, self.tank_2]:
                tank.bullets.update()
                tank.bullets.draw(self.screen)


            pygame.display.flip()
            self.clock.tick(clock)  # Increase the frame rate for smoother rendering

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()





"""
If you want to play the game yourself, set True.
Press s to get the state.

Don't forget to reset it as False, otherwise the training will fail
"""

if False:
    # Initialize Pygame
    pygame.init()

    # Create the game environment
    game = TanksGame()

    run = True

    # Main loop
    while run:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()  

        # first Player
        if keys[pygame.K_LEFT]:
            game.strafe_left(num_tank = 1)
        if keys[pygame.K_RIGHT]:
            game.strafe_right(num_tank = 1)
        if keys[pygame.K_UP]:
            game.move_forward(num_tank = 1)
        if keys[pygame.K_DOWN]:
            game.move_backward(num_tank = 1)
        if keys[pygame.K_l]:
            game.rotate_tank(num_tank = 1, rotation_direction = 'left')
        if keys[pygame.K_m]:
            game.rotate_tank(num_tank = 1, rotation_direction = 'right')
        if keys[pygame.K_k]:
            game.fire_bullet(num_tank = 1)
        

        # second one
        if keys[pygame.K_q]:
            game.strafe_left(num_tank = 2)
        if keys[pygame.K_d]:
            game.strafe_right(num_tank = 2)
        if keys[pygame.K_z]:
            game.move_forward(num_tank = 2)
        if keys[pygame.K_s]:
            game.move_backward(num_tank = 2)
        if keys[pygame.K_a]:
            game.rotate_tank(num_tank = 2, rotation_direction = 'left')
        if keys[pygame.K_e]:
            game.rotate_tank(num_tank = 2, rotation_direction = 'right')
        if keys[pygame.K_f]:
            game.fire_bullet(num_tank = 2)
        
        if keys[pygame.K_g]:
            print('state 1 :, ', game.get_state(num_tank = 1))
            print('state 2 :, ', game.get_state(num_tank = 2))

        # Check for bullet hits
        game.check_bullet_collisions()

        # Render everything
        game.render(rendering=True, clock=300)

        pygame.display.flip()