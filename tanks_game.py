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
TANK_2_SPEED = 3
ROTATION_ANGLE_1 = 1 # But rotates slower
ROTATION_ANGLE_2 = 2
TANK_SIZE = 70
BULLET_DAMAGE = 100
BLOCK_SIZE = 25
LASER_MAX_SIZE = int(np.sqrt(SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2))

background = Background(image_file = BACKGROUND, location = [0,0], width = SCREEN_WIDTH, height = SCREEN_HEIGHT, rendering = RENDERING)

class TanksGame:
    def __init__(self, rendering=RENDERING):
        if rendering:
            pygame.font.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        self.screen_dims = [SCREEN_WIDTH, SCREEN_HEIGHT]
        self.position_1 = [100, rd.randint(100, 600)]
        self.tank_1 = TankPlayer(image_file=TANK_1_IMAGE, location=self.position_1, width=TANK_SIZE, speed=TANK_1_SPEED, rendering=RENDERING)
        self.tank_1.rotate(-30)
        self.position_2 = [SCREEN_WIDTH - 100, rd.randint(100, SCREEN_HEIGHT - 100)]
        self.tank_2 = TankPlayer(image_file=TANK_2_IMAGE, location=self.position_2, width=TANK_SIZE, speed=TANK_2_SPEED, rendering=RENDERING)
        self.tank_2.rotate(150)

        self.last_laser_update = time.time()
        self.laser_update_interval = 0.1  # Adjust this interval based on your needs
        self.cached_laser_distances = [[0, 0, 0], [0, 0, 0]]  # Initial cache for laser distances

        self.middle_block = Block(image_file=CRATE_IMAGE, location=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), width=BLOCK_SIZE, height=BLOCK_SIZE * 2, rendering=RENDERING)

        # Cache radian angles for each tank
        self.tank_1.cached_rad_angle = np.radians(self.tank_1.direction)
        self.tank_2.cached_rad_angle = np.radians(self.tank_2.direction)

    def reset(self):
        self.position_1 = [100, rd.randint(100, 600)]
        self.tank_1 = TankPlayer(image_file=TANK_1_IMAGE, location=self.position_1, width=TANK_SIZE, speed=TANK_1_SPEED, rendering=RENDERING)
        self.tank_1.rotate(rd.randint(-30, 30))
        self.tank_1.cached_rad_angle = np.radians(self.tank_1.direction)
        
        self.position_2 = [SCREEN_WIDTH - 100, rd.randint(100, SCREEN_HEIGHT - 100)]
        self.tank_2 = TankPlayer(image_file=TANK_2_IMAGE, location=self.position_2, width=TANK_SIZE, speed=TANK_2_SPEED, rendering=RENDERING)
        self.tank_2.rotate(rd.randint(150, 210))
        self.tank_2.cached_rad_angle = np.radians(self.tank_2.direction)


        
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
            directions = [0, 20, 340]
            self.cached_laser_distances = [
                [self.cast_laser(direction - self.tank_1.direction, max_distance, 1) for direction in directions],
                [self.cast_laser(direction - self.tank_2.direction, max_distance, 2) for direction in directions]
            ]
            self.last_laser_update = current_time
        return self.cached_laser_distances

    def get_angle_to_opponent(self, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        opponent_position = getattr(self, f'position_{3 - num_tank}')
        position = getattr(self, f'position_{num_tank}')

        # Calculate relative position and angle in one step
        delta_x, delta_y = opponent_position[0] - position[0], opponent_position[1] - position[1]
        angle_to_opponent = np.degrees(math.atan2(delta_y, delta_x))  # Faster than np.arctan2

        # Calculate relative angle with respect to the tank's direction
        relative_angle = (angle_to_opponent - tank.direction + 360) % 360
        return relative_angle


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
            "relative_angle": 180,  # Middle of 0-360 range
        }

        stds = {
            "position": np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2]),  # Assuming range is 0-SCREEN_WIDTH or SCREEN_HEIGHT
            "direction": 180,  # Assuming range 0-360
            "opponent_direction": 180,  # Assuming range 0-360
            "relative_angle": 180,  # Assuming range 0-360
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
            self.standardize(position, means["position"], stds["position"]),
            [self.standardize(direction, means["direction"], stds["direction"])],
            [self.normalize(health, maxs["health"])],
            self.normalize(relative_position, maxs["relative_position"]),
            [self.standardize(opponent_direction, means["opponent_direction"], stds["opponent_direction"])],
            [self.normalize(opponent_health, maxs["opponent_health"])],
            [self.standardize(relative_angle_toward_opponent, means["relative_angle"], stds["relative_angle"])],
            [self.normalize(distance_to_opponent, maxs["distance_to_opponent"])],
            [self.normalize(distance_to_block, maxs["distance_to_block"])],
            [self.normalize(ammo, maxs["ammo"])],
            self.normalize(laser_distances, maxs["laser_distances"]),
            [close_left],
            [in_sight],
            [close_right],
            [is_reloaded],
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

        laser_distances = np.array(self.get_all_laser_distances(LASER_MAX_SIZE)[num_tank - 1])/LASER_MAX_SIZE

        # Reward for reducing the distance to the opponent while maintaining an optimal range
        optimal_distance = 200
        if abs(new_distance_between - optimal_distance) < abs(previous_distance_between - optimal_distance):
            tank.reward += 6
        else:
            tank.reward -= 4

        # Reward for reducing the angle difference with the opponent
        if new_relative_angle_toward_opponent < previous_relative_angle_toward_opponent:
            tank.reward += 2
        else:
            tank.reward -= 2

        # Penalty based on proximity to walls
        distance_to_nearest_wall = min(position[0], SCREEN_WIDTH - position[0], position[1], SCREEN_HEIGHT - position[1])
        wall_proximity_penalty = max(0, (100 - distance_to_nearest_wall) / 100)  # Penalty scales as the tank gets closer to the wall
        tank.reward -= int(wall_proximity_penalty * 10)  # Tune the multiplier as needed

        if self.is_head_against_the_wall(laser_distances):
            tank.reward -= 15  # Penalty for bumping into the wall

        if opponent_tank.was_hit:
            tank.reward += 50  # Large reward for hitting the opponent
            opponent_tank.was_hit = False

        if tank.was_hit:
            tank.reward -= 30  # Penalty for getting hit
            tank.was_hit = False

        if tank.in_line_of_sight:
            tank.reward += 5  # Reward for keeping the opponent in sight

        if tank.on_close_right and rotate_action == 0:
            tank.reward += 2  # Encourage rotating towards the opponent

        if tank.on_close_left and rotate_action == 1:
            tank.reward += 2  # Encourage rotating towards the opponent

        if fire_action == 0 and tank.in_line_of_sight:
            tank.reward += 25  # Reward for firing when the opponent is in sight

        if tank.number_of_ammo == 0:
            tank.reward -= 60  # Penalty for running out of ammo

        # Penalty for staying idle
        if move_action == strafe_action == rotate_action == 2:
            tank.reward -= 10

        if self.lost(tank):
            tank.reward -= 100  # Heavy penalty for losing
            done = True
        elif self.lost(opponent_tank):
            tank.reward += 200  # Large reward for winning
            done = True
        else:
            done = False

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
        laser_angles = [0, 20, 340]
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
            score_text = score_font.render(f"[Tank 1] Current reward : {tank.total_reward} Health = {tank.health} %", True, (0, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.topleft = (10, 10)
        else:
            score_text = score_font.render(f"[Tank 2] Current reward : {tank.total_reward} Health = {tank.health} %", True, (0, 0, 0))
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