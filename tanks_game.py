# tanks_game.py
import math
import pygame
import numpy as np
import random as rd
import time

from tanks_game_objects import Background, TankPlayer, Bullet, Block
from tanks_paths import BACKGROUND, TANK_1_IMAGE, TANK_2_IMAGE, BULLET_IMAGE, CRATE_IMAGE, RENDERING


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
TANK_1_SPEED = 5 # Tank 1 is faster
TANK_2_SPEED = 0
ROTATION_ANGLE_1 = 1 # But rotates slower
ROTATION_ANGLE_2 = 0
TANK_SIZE = 70
BULLET_DAMAGE = 34
BLOCK_SIZE = 1
LASER_MAX_SIZE = int(np.sqrt(SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2))

background = Background(image_file = BACKGROUND, location = [0,0], width = SCREEN_WIDTH, height = SCREEN_HEIGHT, rendering = RENDERING)

class TanksGame:
    def __init__(self, max_steps, rendering=RENDERING):
        if rendering:
            pygame.font.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        self.screen_dims = [SCREEN_WIDTH, SCREEN_HEIGHT]
        self.position_1 = [100, rd.randint(100, 600)]
        self.tank_1 = TankPlayer(image_file=TANK_1_IMAGE, location=self.position_1, width=TANK_SIZE, speed=TANK_1_SPEED, rendering=RENDERING)
        self.tank_1.rotate(0)
        self.position_2 = [SCREEN_WIDTH - 100, rd.randint(100, SCREEN_HEIGHT - 100)]
        self.tank_2 = TankPlayer(image_file=TANK_2_IMAGE, location=self.position_2, width=TANK_SIZE, speed=TANK_2_SPEED, rendering=RENDERING)
        self.tank_2.rotate(0)

        self.last_laser_update = time.time()
        self.laser_update_interval = 0.1  # Adjust this interval based on your needs
        self.cached_laser_distances = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]  # Initial cache for laser distances

        self.middle_block = Block(image_file=CRATE_IMAGE, location=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), width=BLOCK_SIZE, height=BLOCK_SIZE * 2, rendering=RENDERING)

        # Cache radian angles for each tank
        self.tank_1.cached_rad_angle = np.radians(self.tank_1.direction)
        self.tank_2.cached_rad_angle = np.radians(self.tank_2.direction)

        self.current_step = 0
        self.max_steps = max_steps


    def reset(self):
        self.position_1 = [100, rd.randint(100, 600)]
        self.tank_1 = TankPlayer(image_file=TANK_1_IMAGE, location=self.position_1, width=TANK_SIZE, speed=TANK_1_SPEED, rendering=RENDERING)
        self.tank_1.rotate(0)
        self.tank_1.cached_rad_angle = np.radians(self.tank_1.direction)
        
        self.position_2 = [SCREEN_WIDTH - 100, rd.randint(100, SCREEN_HEIGHT - 100)]
        self.tank_2 = TankPlayer(image_file=TANK_2_IMAGE, location=self.position_2, width=TANK_SIZE, speed=TANK_2_SPEED, rendering=RENDERING)
        self.tank_2.rotate(0)
        self.tank_2.cached_rad_angle = np.radians(self.tank_2.direction)

        self.current_step = 0


        
    def rotate_tank(self, num_tank, rotation_direction):
        direction_of_rotation = -1 if rotation_direction == 'right' else 1
        tank = getattr(self, f'tank_{num_tank}')
        rotation_angle = ROTATION_ANGLE_1 if num_tank == 1 else ROTATION_ANGLE_2
        tank.rotate(rotation_angle * direction_of_rotation)
        tank.cached_rad_angle = np.radians(tank.direction)  # Update cached radian angle

    def calculate_movement(self, num_tank, angle_offset=0, slowdown=1):
        tank = getattr(self, f'tank_{num_tank}')
        rad_angle = tank.cached_rad_angle + angle_offset * (np.pi / 180)
        dx = tank.speed * np.cos(rad_angle) * slowdown
        dy = -tank.speed * np.sin(rad_angle) * slowdown  # Negative because Pygame's y-axis is flipped
        return dx, dy

    def move(self, num_tank=1, dx=0, dy=0):
        position = getattr(self, f'position_{num_tank}')
        position[0] += dx
        position[1] += dy

    def strafe_left(self, num_tank):
        dx, dy = self.calculate_movement(num_tank, angle_offset=90, slowdown=0.3)
        self.move(num_tank, dx, dy)

    def strafe_right(self, num_tank):
        dx, dy = self.calculate_movement(num_tank, angle_offset=-90, slowdown=0.3)
        self.move(num_tank, dx, dy)

    def move_forward(self, num_tank):
        dx, dy = self.calculate_movement(num_tank)
        self.move(num_tank, dx, dy)

    def move_backward(self, num_tank):
        dx, dy = self.calculate_movement(num_tank, slowdown=0.6)
        self.move(num_tank, -dx, -dy)

    
    def update_tank_position(self):
        for i in [1, 2]:
            tank = getattr(self, f'tank_{i}')
            position = getattr(self, f'position_{i}')  

            tank_width = TANK_SIZE //2 
            position[0] = np.clip(position[0], tank_width, SCREEN_WIDTH - tank_width) # restrain position[0] between 0 and SCREEN_WIDTH (minus half of the tank width)
            position[1] = np.clip(position[1], tank_width, SCREEN_HEIGHT - tank_width)

            x_top_left, y_top_left = self.middle_block.rect.topleft
            x_bottom_right, y_bottom_right = self.middle_block.rect.bottomright

            # Clip position to avoid the block, clip is obviously not the best choice but I found it more visual and functional
            if x_top_left <= position[0] <= x_bottom_right and position[1] <= y_top_left:
                position[1] = np.clip(position[1], tank_width, y_top_left - tank_width)

            if x_top_left <= position[0] <= x_bottom_right and position[1] >= y_bottom_right:
                position[1] = np.clip(position[1], y_bottom_right + tank_width, SCREEN_HEIGHT - tank_width)

            if y_top_left <= position[1] <= y_bottom_right and position[0] <= x_top_left:
                position[0] = np.clip(position[0], tank_width, x_top_left - tank_width)

            if y_top_left <= position[1] <= y_bottom_right and position[0] >= x_bottom_right:
                position[0] = np.clip(position[0], x_bottom_right + tank_width, SCREEN_WIDTH - tank_width)
            
            # Update tank position
            tank.rect.center = position[0] , position[1]


    def fire_bullet(self, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position = getattr(self, f'position_{num_tank}')
        x, y = position

        current_time = time.time()
        if tank.health > 0 and tank.number_of_ammo > 0 and tank.check_cooldown(current_time):
            bullet = Bullet(
                image_file=BULLET_IMAGE,
                location=(x, y),
                direction=tank.direction,
                width=10,
                screen_dims=(SCREEN_WIDTH, SCREEN_HEIGHT),
                rendering=RENDERING,
                game_instance=self,
                block=self.middle_block
            )
            tank.number_of_ammo -= 1
            tank.bullets.add(bullet) if RENDERING else tank.bullets.append(bullet)
            tank.cooldown = current_time

    def check_bullet_collisions(self):
        for i, tank in enumerate([self.tank_1, self.tank_2]):
            opponent_tank = self.tank_1 if i == 1 else self.tank_2
            bullets = tank.bullets if RENDERING else tank.bullets[:]
            for bullet in bullets:
                if bullet.check_collision(opponent_tank.rect):
                    self.handle_collision(bullet, opponent_tank, 2 - i)
                elif bullet.block and bullet.rect.colliderect(bullet.block.rect):
                    self.handle_block_collision(bullet)

    def update_bullets(self):
        for tank in [self.tank_1, self.tank_2]:
            bullets = tank.bullets if RENDERING else tank.bullets[:]
            for bullet in bullets:
                bullet.update()
                # Update these conditions to use the custom Rect properties
                if not (0 < bullet.rect.left < self.screen_dims[0]) or not (0 < bullet.rect.top < self.screen_dims[1]):
                    tank.bullets.remove(bullet)


    def handle_block_collision(self, bullet):
        bullet.kill()

    def handle_collision(self, bullet, tank, num_tank):
        tank.health -= BULLET_DAMAGE
        bullet.kill()
        tank.was_hit = True
        print(f"Tank {num_tank} was hit!")

    def cast_laser(self, direction, max_distance, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position = getattr(self, f'position_{num_tank}')
        opponent_tank = getattr(self, f'tank_{3 - num_tank}')
        x, y = position
        angle_rad = math.radians(direction)

        for distance in range(1, max_distance + 1, 5):  # Use approximation of 5
            ray_x = x + distance * math.cos(angle_rad)
            ray_y = y + distance * math.sin(angle_rad)

            # Check if ray hits border or middle block
            if not (0 <= ray_x < SCREEN_WIDTH and 0 <= ray_y < SCREEN_HEIGHT):
                return distance
            if self.middle_block.rect.collidepoint(ray_x, ray_y):
                return distance

            # Update tank's laser detection status
            compensated_laser_direction = (direction + tank.direction) % 360
            if compensated_laser_direction == 0:
                tank.in_line_of_sight = False
            elif compensated_laser_direction == 20:
                tank.on_close_right = False
            elif compensated_laser_direction == 340:
                tank.on_close_left = False

            if opponent_tank.rect.collidepoint(ray_x, ray_y):
                if compensated_laser_direction == 0:
                    tank.in_line_of_sight = True
                elif compensated_laser_direction == 20:
                    tank.on_close_right = True
                elif compensated_laser_direction == 340:
                    tank.on_close_left = True
                return distance

        return max_distance

    def get_all_laser_distances(self, max_distance):
        current_time = time.time()
        if current_time - self.last_laser_update > self.laser_update_interval:
            directions = [0, 20, 90, 270, 340]
            self.cached_laser_distances = [
                [self.cast_laser(direction - self.tank_1.direction, max_distance, 1) for direction in directions],
                [self.cast_laser(direction - self.tank_2.direction, max_distance, 2) for direction in directions]
            ]
            self.last_laser_update = current_time
        return self.cached_laser_distances

    def get_angle_to_opponent(self, num_tank):
        # Get tank and opponent positions
        tank = getattr(self, f'tank_{num_tank}')
        opponent_position = getattr(self, f'position_{3 - num_tank}')
        position = getattr(self, f'position_{num_tank}')

        # Calculate the vector from tank to opponent
        vector_to_opponent = np.array(opponent_position) - np.array(position)

        # Normalize the vector
        vector_to_opponent = vector_to_opponent / np.linalg.norm(vector_to_opponent)

        # Get the facing direction of the tank as a unit vector
        rad_angle = np.radians(tank.direction)
        facing_vector = np.array([np.cos(rad_angle), -np.sin(rad_angle)])  # -sin because of the flipped y-axis in Pygame

        # Calculate the angle between the facing vector and the vector to the opponent
        dot_product = np.dot(facing_vector, vector_to_opponent)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to avoid numerical issues

        # Determine the sign of the angle by checking the cross product
        cross_product = facing_vector[0] * vector_to_opponent[1] - facing_vector[1] * vector_to_opponent[0]
        if cross_product < 0:
            angle = -angle  # Adjust sign based on relative orientation

        return angle


    def standardize(self, value, mean, std):
        return np.where(std != 0, (value - mean) / std, value)

    def normalize(self, value, max_value):
        return np.where(max_value != 0, value / max_value, value)


    def get_standardization_parameters(self):
        # Define means, standard deviations (stds), and max values for each feature.
        means = {
            "position": np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2]),  # Assuming the middle of the screen
            "direction": 180,  # Middle of 0-360 range
            "opponent_direction": 180,  # Middle of 0-360 range
        }

        stds = {
            "position": np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2]),  # Assuming range is 0-SCREEN_WIDTH or SCREEN_HEIGHT
            "direction": 180,  # Assuming range 0-360
            "opponent_direction": 180,  # Assuming range 0-360
            "relative_angle": np.pi,
        }

        maxs = {
            "relative_position": np.array([SCREEN_WIDTH, SCREEN_HEIGHT]),  # Maximum relative position in each axis
            "health": 100,  # Maximum health
            "opponent_health": 100,  # Maximum opponent health
            "ammo": 1000,  # Example maximum ammo, adjust as needed
            "laser_distances": LASER_MAX_SIZE,  # Maximum laser distance
            "distance_to_opponent": np.linalg.norm([SCREEN_WIDTH, SCREEN_HEIGHT]),  # Maximum diagonal distance
            "distance_to_block": np.linalg.norm([SCREEN_WIDTH, SCREEN_HEIGHT]),  # Maximum diagonal distance to block
        }

        return means, stds, maxs


    def get_state(self, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        opponent_tank = getattr(self, f'tank_{3 - num_tank}')
        position = getattr(self, f'position_{num_tank}')
        opponent_position = getattr(self, f'position_{3 - num_tank}')

        health, direction = tank.health, tank.direction
        opponent_health, opponent_direction = opponent_tank.health, opponent_tank.direction
        relative_angle_toward_opponent = self.get_angle_to_opponent(num_tank)
        ammo, in_sight = tank.number_of_ammo, tank.in_line_of_sight
        close_right, close_left, is_reloaded = tank.on_close_right, tank.on_close_left, tank.check_cooldown(time.time())

        distance_to_opponent = np.sqrt((opponent_position[0] - position[0])**2 + (opponent_position[1] - position[1])**2)
        block_center = self.middle_block.rect.center
        distance_to_block = np.sqrt((position[0] - block_center[0])**2 + (position[1] - block_center[1])**2)

        relative_position = np.array([opponent_position[0] - position[0], opponent_position[1] - position[1]])
        angle_rad = -math.radians(direction)
        rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]])
        relative_position = np.dot(rotation_matrix, relative_position)

        laser_distances = np.array(self.get_all_laser_distances(LASER_MAX_SIZE)[num_tank - 1])

        means, stds, maxs = self.get_standardization_parameters()

        state = np.concatenate([
            self.standardize(position, means["position"], stds["position"]),          # (2) 
            [self.standardize(direction, means["direction"], stds["direction"])],     # (1)
            [self.normalize(health, maxs["health"])],                                 # (1)
            self.normalize(relative_position, maxs["relative_position"]),             # (2)
            [self.standardize(opponent_direction, means["opponent_direction"], stds["opponent_direction"])], # (1)
            [self.normalize(opponent_health, maxs["opponent_health"])],               # (1)
            [self.normalize(relative_angle_toward_opponent, stds["relative_angle"])], # (1)
            [self.normalize(distance_to_opponent, maxs["distance_to_opponent"])],     # (1)
            [self.normalize(distance_to_block, maxs["distance_to_block"])],           # (1)
            [self.normalize(ammo, maxs["ammo"])],                                     # (1)
            self.normalize(laser_distances, maxs["laser_distances"]),                 # (5)
            [close_left],                                                             # (1)
            [in_sight],                                                               # (1)
            [close_right],                                                            # (1)
            [is_reloaded],                                                            # (1)
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

        previous_distance_between = np.linalg.norm(np.array(position) - np.array(opponent_position))
        previous_angle_to_opponent = abs(self.get_angle_to_opponent(num_tank = num_tank))
        
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
        new_angle_to_opponent = abs(self.get_angle_to_opponent(num_tank = num_tank))

        self.update_bullets()           # Ensure bullets are updated every step
        self.check_bullet_collisions()  # Ensure collisions are checked every step

        laser_distances = np.array(self.get_all_laser_distances(LASER_MAX_SIZE)[num_tank - 1])/LASER_MAX_SIZE

        # Reward for reducing the distance to the opponent while maintaining an optimal range
        optimal_distance = 400

        if new_distance_between < previous_distance_between and new_distance_between > optimal_distance :
            tank.reward += 2
        elif new_distance_between > optimal_distance :
            tank.reward -= 1

        if new_angle_to_opponent <= previous_angle_to_opponent :
            tank.reward += 2
        elif new_angle_to_opponent > 0.1:
            tank.reward -= 1

        tank.reward -= 5 * new_angle_to_opponent

        if self.is_head_against_the_wall(laser_distances):
            tank.reward -= 5  # Penalty for bumping into the wall

        if opponent_tank.was_hit:
            tank.reward += 100  # Large reward for hitting the opponent
            opponent_tank.was_hit = False

        if tank.was_hit:
            tank.reward -= 30  # Penalty for getting hit
            tank.was_hit = False

        if tank.in_line_of_sight:
            tank.reward += 0.2  # Reward for keeping the opponent in sight

        if tank.number_of_ammo == 0:
            tank.reward -= 60  # Penalty for running out of ammo

        if self.lost(tank):
            tank.reward -= 1_000  # Heavy penalty for losing
            done = True
        elif self.lost(opponent_tank):
            tank.reward += 2_000  # Large reward for winning
            done = True
        elif self.current_step > self.max_steps:
            tank.reward -= 1_000  # Heavy penalty for running out of time
            done = True
        else:
            done = False

        self.current_step += 1
        tank.total_reward += tank.reward
        return self.get_state(num_tank), tank.reward, done, {}

    def lost(self, tank):
        return tank.health <= 0

    def draw_hitboxes(self, draw=False):
        if draw:
            # Draw hitboxes around the tanks
            pygame.draw.rect(self.screen, (161, 155, 88), self.tank_1.rect, 2)  # Desert hitbox for tank 1
            pygame.draw.rect(self.screen, (41, 79, 23), self.tank_2.rect, 2)    # Green hitbox for tank 2

    def draw_laser(self, tank, position, laser_distances, num_tank):
        laser_angles = [0, 20, 90, 270, 340]
        for i, direction in enumerate(laser_angles):
            laser_angle = direction  # save laser angle value
            direction -= tank.direction
            distance = laser_distances[i]
            angle_rad = math.radians(direction)
            start_pos = (position[0], position[1])
            end_pos = (
                position[0] + distance * math.cos(angle_rad),
                position[1] + distance * math.sin(angle_rad),
            )
            color = (161, 155, 88) if num_tank == 1 else (41, 79, 23)
            if (laser_angle == 0 and tank.in_line_of_sight) or (laser_angle == 20 and tank.on_close_right) or (laser_angle == 340 and tank.on_close_left):
                color = (255, 0, 0)
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
    
    def draw_text(self, tank, num_tank, epsilon=None):
        score_font = pygame.font.Font(pygame.font.get_default_font(), 14)
        background_color = (161, 155, 88)  # White semi-transparent background

        if num_tank == 1:
            score_text = score_font.render(f"[Tank 1] Current reward : {tank.total_reward:.2f} Health = {tank.health}%", True, (0, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.topleft = (10, 10)
        else:
            score_text = score_font.render(f"[Tank 2] Current reward : {tank.total_reward:.2f} Health = {tank.health}%", True, (0, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.topright = (SCREEN_WIDTH - 10, 10)

        pygame.draw.rect(self.screen, background_color, score_rect.inflate(10, 10))
        self.screen.blit(score_text, score_rect)

        if epsilon is not None:
            epsilon_text = score_font.render(f"Randomness: {epsilon * 100:.1f}%", True, (0, 0, 0))
            epsilon_rect = epsilon_text.get_rect()
            epsilon_rect.topleft = (SCREEN_WIDTH // 2 - 50, 10)
            
            pygame.draw.rect(self.screen, background_color, epsilon_rect.inflate(10, 10))
            self.screen.blit(epsilon_text, epsilon_rect)

    def render(self, rendering, clock, epsilon=0):
        if rendering:
            self.screen.fill([255, 255, 255])
            self.screen.blit(background.image, background.rect)

            # Draw the middle block
            self.screen.blit(self.middle_block.image, self.middle_block.rect)

            # Draw the tanks
            self.update_tank_position()
            if self.tank_1.health > 0:
                self.screen.blit(self.tank_1.image, self.tank_1.rect)
            if self.tank_2.health > 0:
                self.screen.blit(self.tank_2.image, self.tank_2.rect)

            self.draw_hitboxes(False)

            max_distance = LASER_MAX_SIZE
            laser_distances = self.get_all_laser_distances(max_distance)

            # Draw lasers if necessary
            self.draw_laser(tank=self.tank_1, position=self.position_1, laser_distances=laser_distances[0], num_tank=1)
            self.draw_laser(tank=self.tank_2, position=self.position_2, laser_distances=laser_distances[1], num_tank=2)

            self.draw_text(tank=self.tank_1, num_tank=1, epsilon=epsilon)
            self.draw_text(tank=self.tank_2, num_tank=2)

            # Update bullets
            for tank in [self.tank_1, self.tank_2]:
                tank.bullets.update()
                tank.bullets.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(clock)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()





"""
If you want to play the game yourself, set True.
Press s to get the state.

Don't forget to reset it as False, otherwise the training will fail
"""

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()

    # Create the game environment
    game = TanksGame()

    run = True

    # Main loop
    while run:

        keys = pygame.key.get_pressed()  

        for event in pygame.event.get():
            if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                run = False

        # first Player
        if keys[pygame.K_q]:
            game.strafe_left(num_tank = 1)
        if keys[pygame.K_d]:
            game.strafe_right(num_tank = 1)
        if keys[pygame.K_z]:
            game.move_forward(num_tank = 1)
        if keys[pygame.K_s]:
            game.move_backward(num_tank = 1)
        if keys[pygame.K_a]:
            game.rotate_tank(num_tank = 1, rotation_direction = 'left')
        if keys[pygame.K_e]:
            game.rotate_tank(num_tank = 1, rotation_direction = 'right')
        if keys[pygame.K_f]:
            game.fire_bullet(num_tank = 1)
        # second one
        if keys[pygame.K_LEFT]:
            game.strafe_left(num_tank = 2)
        if keys[pygame.K_RIGHT]:
            game.strafe_right(num_tank = 2)
        if keys[pygame.K_UP]:
            game.move_forward(num_tank = 2)
        if keys[pygame.K_DOWN]:
            game.move_backward(num_tank = 2)
        if keys[pygame.K_l]:
            game.rotate_tank(num_tank = 2, rotation_direction = 'left')
        if keys[pygame.K_m]:
            game.rotate_tank(num_tank = 2, rotation_direction = 'right')
        if keys[pygame.K_k]:
            game.fire_bullet(num_tank = 2)
        
        if keys[pygame.K_r]:
            game.reset()
        
        if keys[pygame.K_g]:
            state_1 = game.get_state(num_tank=1)
            state_2 = game.get_state(num_tank=2)

            print("\n--- Tank 1 State ---")
            print(f"Position: {state_1[:2]}")
            print(f"Direction: {state_1[2]}")
            print(f"Health: {state_1[3]}")
            print(f"Relative Position to Opponent: {state_1[4:6]}")
            print(f"Opponent Direction: {state_1[6]}")
            print(f"Opponent Health: {state_1[7]}")
            print(f"Relative Angle to Opponent: {state_1[8]}")
            print(f"Distance to Opponent: {state_1[9]}")
            print(f"Distance to Block: {state_1[10]}")
            print(f"Ammo: {state_1[11]}")
            print(f"Laser Distances: {state_1[12:17]}")
            print(f"Close Left: {state_1[17]}")
            print(f"In Line of Sight: {state_1[18]}")
            print(f"Close Right: {state_1[19]}")
            print(f"Is Reloaded: {state_1[20]}")

            print("\n--- Tank 2 State ---")
            print(f"Position: {state_2[:2]}")
            print(f"Direction: {state_2[2]}")
            print(f"Health: {state_2[3]}")
            print(f"Relative Position to Opponent: {state_2[4:6]}")
            print(f"Opponent Direction: {state_2[6]}")
            print(f"Opponent Health: {state_2[7]}")
            print(f"Relative Angle to Opponent: {state_2[8]}")
            print(f"Distance to Opponent: {state_2[9]}")
            print(f"Distance to Block: {state_2[10]}")
            print(f"Ammo: {state_2[11]}")
            print(f"Laser Distances: {state_2[12:17]}")
            print(f"Close Left: {state_2[17]}")
            print(f"In Line of Sight: {state_2[18]}")
            print(f"Close Right: {state_2[19]}")
            print(f"Is Reloaded: {state_2[20]}")

            time.sleep(0.1)

        # Check for bullet hits
        game.check_bullet_collisions()

        # Render everything
        game.render(rendering=True, clock=300)

        pygame.display.flip()

# TODO : ADD power-ups and heals