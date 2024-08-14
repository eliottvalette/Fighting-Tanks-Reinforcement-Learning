BACKGROUND = 'Tanks/images/empty_board.jpg'
TANK_1_IMAGE = 'Tanks/images/red_tank.png'
TANK_2_IMAGE = 'Tanks/images/green_tank.png'
TANK_1_WEIGHTS = 'Tanks/Agents/trained_agent_1.pth'
TANK_2_WEIGHTS = 'Tanks/Agents/trained_agent_2.pth'
BULLET_IMAGE = 'Tanks/images/bullet.png'



# tanks_model.py
import torch.nn as nn

class TanksModel(nn.Module):
    def __init__(self, state_size, action_sizes):
        super(TanksModel, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
        )
        
        # Separate output layers for each action space
        self.movement_layer = nn.Linear(32, action_sizes[0])
        self.rotation_layer = nn.Linear(32, action_sizes[1])
        self.strafe_layer = nn.Linear(32, action_sizes[2])
        self.fire_layer = nn.Linear(32, action_sizes[3])
    
    def forward(self, state):
        x = self.shared_net(state)
        q_values_movement = self.movement_layer(x)
        q_values_rotation = self.rotation_layer(x)
        q_values_strafe = self.strafe_layer(x)
        q_values_fire = self.fire_layer(x)
        
        return q_values_movement, q_values_rotation, q_values_strafe, q_values_fire




# tanks_game_objects.py
import pygame
import numpy as np
import random as rd
import time

BULLET_COOLDOWN_TIME = 0.2
BULLET_SPEED = 30
TANK_HEALTH = 100
TANK_AMMO = 1000


class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location, width, height):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(image_file)
        self.image = pygame.transform.scale(self.image, (width, height))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

class TankPlayer(pygame.sprite.Sprite):
    def __init__(self, image_file, location, width, speed = 1):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = pygame.image.load(image_file).convert_alpha()
        self.original_image = pygame.transform.scale(self.original_image, (2 * width, width))
        self.image = self.original_image.copy() # Create a copy for rotation
        self.rect = self.image.get_rect()
        self.direction = 0
        self.rect.center = location
        self.speed = speed
        self.cooldown = time.time()
        self.bullets = pygame.sprite.Group() # Like Empty list but for Classes
        self.in_line_of_sight = False
        self.on_close_right   = False
        self.on_close_left    = False
        self.health = TANK_HEALTH
        self.number_of_ammo = TANK_AMMO
        self.reward = 0
        self.total_reward = 0
        
    def rotate(self, angle):
        self.direction += angle
        self.direction %= 360  # Keep the angle within 0-359 degrees
        self.image = pygame.transform.rotate(self.original_image, self.direction)
        self.rect = self.image.get_rect(center=self.rect.center)
    
    def check_cooldown(self, current_time):
        is_reloaded = current_time - self.cooldown > BULLET_COOLDOWN_TIME
        if is_reloaded : # bullet will be shot, so reset cooldwon
            self.cooldown = time.time()
        return is_reloaded

class Bullet(pygame.sprite.Sprite):
    def __init__(self, image_file, location, direction, width, screen_dims, block=None):
        pygame.sprite.Sprite.__init__(self)
        self.original_image = pygame.image.load(image_file).convert_alpha()
        self.original_image = pygame.transform.scale(self.original_image, (int(2.5 * width), width))
        self.image = self.original_image
        self.rect = self.image.get_rect()
        self.rect.center = location
        self.speed = BULLET_SPEED
        self.direction = direction
        self.screen_dims = screen_dims
        self.block = block  # Reference to the block in the arena

        # Calculate dx and dy based on direction angle
        self.dx = self.speed * np.cos(np.radians(direction))
        self.dy = - self.speed * np.sin(np.radians(direction)) # egative for the same reason

        self.image = pygame.transform.rotate(self.original_image, self.direction)


    def update(self):
        rad_angle = np.radians(self.direction)
        self.rect.x += int(self.speed * np.cos(rad_angle))
        self.rect.y -= int(self.speed * np.sin(rad_angle))  # Negative because Pygame's y-axis is flipped

    def check_collision(self, target_rect):
        return self.rect.colliderect(target_rect)

class Block(pygame.sprite.Sprite):
    def __init__(self, location, width, height):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((width, height))
        self.image.fill((128, 128, 128))  # Grey color for the block
        self.rect = self.image.get_rect()
        self.rect.center = location





# tanks_agent.py
import numpy as np
from collections import deque
import random as rd
import torch
import torch.nn as nn
import torch.optim as optim

class TanksAgent(nn.Module):
    def __init__(self, state_size, action_sizes, gamma, learning_rate, device, load_model = False):
        super(TanksAgent, self).__init__()
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.device = device

        self.memory = deque(maxlen=10_000)
        self.batch_size = 256

        self.model = self.build_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.load_model = load_model

    def build_model(self):
        return TanksModel(self.state_size, self.action_sizes)

    def forward(self, state):
        state = state.to(self.device)
        return self.model(state)

    def get_actions(self, state, epsilon):
        if rd.random() <= epsilon:
            return [rd.randint(0,action_size - 1) for action_size in self.action_sizes]
        
        state = torch.FloatTensor(state)

        # Unpack the Q-values tuple returned by the model
        q_values_movement, q_values_rotation, q_values_strafe, q_values_fire = self.forward(state)

        q_values = [q_values_movement, q_values_rotation, q_values_strafe, q_values_fire]

        actions = []
        for i in range(len(self.action_sizes)):
            q_values_for_action = q_values[i]  # Extract Q-values for this particular action
            best_action = torch.argmax(q_values_for_action).item()
            actions.append(best_action)
        
        return actions
    
    def train_model(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Move to device
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        action = torch.LongTensor(action).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        q_values = self.forward(state)
        next_q_values = self.forward(next_state)

        target_q_values = reward.repeat(len(self.action_sizes))

        # Calculate current Q-value for the taken actions
        current_q_values = torch.stack([q_values[i][action[i]] for i in range(len(self.action_sizes))])

        for i in range(len(self.action_sizes)):
            max_next_q_value = torch.max(next_q_values[i])  # Max Q-value for the next state in this action space
            target_q_values += (1 - done) * self.gamma * max_next_q_value

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = rd.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.train_model(state, action, reward, next_state, done)




# tanks_game.py
import math
import pygame
import numpy as np
import random as rd
import time

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
ROTATION_ANGLE = 4
TANK_SPEED = 10
TANK_SIZE = 50
BLOCK_SIZE = 100
LASER_MAX_SIZE = int(np.sqrt(SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2))
background = Background(image_file = BACKGROUND, location = [0,0], width = SCREEN_WIDTH, height = SCREEN_HEIGHT)

class TanksGame:
    def __init__(self):
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen_dims = [SCREEN_WIDTH, SCREEN_HEIGHT]
        self.clock  = pygame.time.Clock()
        self.position_1 = [100, 100]
        self.tank_1 = TankPlayer(image_file = TANK_1_IMAGE, location = self.position_1, width = TANK_SIZE, speed = TANK_SPEED)
        self.tank_1.rotate(- 30) 
        self.position_2 = [SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100 ]
        self.tank_2 = TankPlayer(image_file = TANK_2_IMAGE, location = self.position_2, width = TANK_SIZE, speed = TANK_SPEED)
        self.tank_2.rotate(150) 

        self.middle_block = Block(location=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2), width=BLOCK_SIZE, height=BLOCK_SIZE * 2)

    def reset(self):
        self.position_1 = [100, 100]
        self.tank_1 = TankPlayer(image_file = TANK_1_IMAGE, location = self.position_1, width = 50, speed = 3)
        self.tank_1.rotate(- 30) 
        self.position_2 = [SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100 ]
        self.tank_2 = TankPlayer(image_file = TANK_2_IMAGE, location = self.position_2, width = 50, speed = 3)
        self.tank_2.rotate(150) 


        
    def rotate_tank(self, num_tank, rotation_direction):
        if rotation_direction == 'right':
            angle = - ROTATION_ANGLE
        else :
            angle = ROTATION_ANGLE
        if num_tank == 1:
            self.tank_1.rotate(angle)
        else:
            self.tank_2.rotate(angle)

    def calculate_movement(self, num_tank, angle_offset=0, slowdown= 1):
        tank = getattr(self, f'tank_{num_tank}')
        rad_angle =  np.radians(tank.direction + angle_offset)
        dx = tank.speed * np.cos(rad_angle) * slowdown
        dy = - tank.speed * np.sin(rad_angle) * slowdown  # Negative because Pygame's y-axis is flipped
        return dx, dy

    def is_walking_on_walls(self, tank, dx, dy):
        new_rect_x = tank.rect.copy()
        new_rect_y = tank.rect.copy()
        new_rect_x.x += dx
        new_rect_y.y += dy
        
        collide_by_x = new_rect_x.colliderect(self.middle_block.rect)
        collide_by_y = new_rect_y.colliderect(self.middle_block.rect)

        return [collide_by_x, collide_by_y]



    def move(self, num_tank = 1, dx = 0, dy = 0):
        if num_tank == 1 :
          is_walking_on_walls = self.is_walking_on_walls(self.tank_1, dx, dy)
          if not is_walking_on_walls[0] :
            self.position_1[0] += dx
          if not is_walking_on_walls[1] :
            self.position_1[1] += dy
        elif num_tank == 2 :
          is_walking_on_walls = self.is_walking_on_walls(self.tank_2, dx, dy)
          if not is_walking_on_walls[0] :
            self.position_2[0] += dx
          if not is_walking_on_walls[1] :
            self.position_2[1] += dy 

    def strafe_left(self, num_tank):
        dx, dy = self.calculate_movement(num_tank, angle_offset=90, slowdown = 0.3) # don't strafe as fast as forward movement
        self.move(num_tank, dx, dy)

    def strafe_right(self, num_tank):
        dx, dy = self.calculate_movement(num_tank, angle_offset=-90, slowdown = 0.3) # don't strafe as fast as forward movement
        self.move(num_tank, dx, dy)

    def move_forward(self, num_tank):
        dx, dy = self.calculate_movement(num_tank)
        self.move(num_tank, dx, dy)

    def move_backward(self, num_tank):
        dx, dy = self.calculate_movement(num_tank, slowdown = 0.6) # don't go backward as fast as forward movement
        self.move(num_tank, -dx, -dy)
    
    def update_tank_position(self):
        for i in [1, 2]:
            tank = getattr(self, f'tank_{i}')
            position = getattr(self, f'position_{i}')  

            position[0] = np.clip(position[0], 30, SCREEN_WIDTH - 30)# restrain position[0] between 0 and SCREEN_WIDTH
            position[1] = np.clip(position[1], 30, SCREEN_HEIGHT - 30)

            x_pos = position[0] 
            y_pos = position[1]

            # Mettre à jour la position du rectangle du tank
            tank.rect.center = x_pos, y_pos


    def fire_bullet(self, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position =  getattr(self, f'position_{num_tank}')
        x, y = position

        direction = tank.direction
        is_reloaded = tank.check_cooldown(current_time = time.time())
            

        if is_reloaded and tank.health  > 0: 

            bullet = Bullet(image_file=BULLET_IMAGE, location=(x, y), 
                            direction=direction, width=10, screen_dims=(SCREEN_WIDTH, SCREEN_HEIGHT),
                            block= self.middle_block)
            
            tank.number_of_ammo -= 1
            tank.bullets.add(bullet)
    
    def check_bullet_collisions(self):
        tanks = [self.tank_1, self.tank_2]
        for i, tank in enumerate(tanks):
            for bullet in tank.bullets:
                opponent_tank = self.tank_1 if i == 1 else self.tank_2
                if bullet.check_collision(opponent_tank.rect):
                    self.handle_collision(bullet, opponent_tank, i + 1)
                elif bullet.block and bullet.rect.colliderect(bullet.block.rect):
                    self.handle_block_collision(bullet)
    
    def update_bullets(self):
        for tank in [self.tank_1, self.tank_2]:
            for bullet in tank.bullets:
                bullet.update()
                if bullet.rect.x < 0 or bullet.rect.x > self.screen_dims[0] or bullet.rect.y < 0 or bullet.rect.y > self.screen_dims[1]:
                    tank.bullets.remove(bullet)


    def handle_block_collision(self, bullet):
        bullet.kill()
        print("Bullet hit the block!")
        
    def handle_collision(self, bullet, tank, num_tank):
        tank.health -= 100
        bullet.kill()
        print(f"Tank {num_tank} was hit!")
    


    def cast_laser(self, direction, max_distance, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position =  getattr(self, f'position_{num_tank}')

        opponent_tank = getattr(self, f'tank_{3-num_tank}')
        x, y = position
        angle_rad = math.radians(direction)

        for distance in range(1, max_distance + 1):
            ray_x = x + distance * math.cos(angle_rad)
            ray_y = y + distance * math.sin(angle_rad)

            # Check if the ray hits a border (assuming a rectangular map)
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




    def get_state(self, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position = np.array(getattr(self, f'position_{num_tank}'))
        health = tank.health
        direction = tank.direction

        opponent_tank = getattr(self, f'tank_{3 - num_tank}')
        opponent_position = np.array(getattr(self, f'position_{3 - num_tank}'))
        opponent_health = opponent_tank.health
        opponent_direction = opponent_tank.direction
        
        ammo = tank.number_of_ammo
        in_sight = tank.in_line_of_sight
        close_right = tank.on_close_right
        close_left = tank.on_close_left

        is_reloaded = tank.check_cooldown(current_time = time.time())


        # Get laser distances and convert to a NumPy array for division
        laser_distances = np.array(self.get_all_laser_distances(800)[num_tank - 1])

        state = np.concatenate([
            position / 800,               # Position (Size : 2)
            [direction / 360] ,           # Direction (Size : 1)
            [health / 100] ,              # Health (Size : 1)

            opponent_position / 800,      # 0pponent position (Size : 2)
            [opponent_direction / 360],   # Opponent Direction (Size : 1)
            [opponent_health / 100] ,     # Opponent Health (Size : 1)

            [ammo / 10] ,                 # Ammo count (Size : 1)
            laser_distances / 800,        # Laser distances (Size : 10)
            [close_left],                 # Close Right Boolean (Size : 1)
            [in_sight],                   # In Sight Boolean (Size : 1)
            [close_right],                # Close Right Boolean (Size : 1)
            [is_reloaded],                # Is Reloaded Boolean (Size : 1)
            
        ])

        return state

    def step(self, actions, num_tank):
        tank = getattr(self, f'tank_{num_tank}')
        position = getattr(self, f'position_{num_tank}')
        opponent_tank = getattr(self, f'tank_{3 - num_tank}')
        opponent_position = getattr(self, f'position_{3 - num_tank}')

        tank.reward=0

        move_action, rotate_action, strafe_action, fire_action = actions

        previous_distance_between_them = np.linalg.norm(np.array(position) - np.array(opponent_position))
        
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

        # Always run the update and collision logic
        self.update_bullets()  # Ensure bullets are updated every step
        self.check_bullet_collisions()  # Ensure collisions are checked every step

        if self.lost(tank):
            tank.reward -= 400  # Heavy penalty for losing
            done = True
        elif self.lost(opponent_tank):
            tank.reward += 1000  # Big reward for winning
            done = True
        else:
            done = False
            if new_distance_between <  previous_distance_between_them :
                tank.reward += 3
            else :
                tank.reward -= 3

            if opponent_tank.health < 100:
                tank.reward += 500  # Reward for hitting the opponent

            if tank.health < 100:
                tank.reward -= 100  # Penalty for getting hit

            if tank.in_line_of_sight:
                tank.reward += 10  # Encourage aiming at the opponent
            
            if tank.on_close_right or tank.on_close_left:
                tank.reward += 6  # Encourage aiming close to the opponent

            if fire_action == 0 and tank.in_line_of_sight:
                tank.reward += 20 # reward fire action in sight
            
            elif fire_action == 0 and (tank.on_close_right or tank.on_close_left):
                tank.reward += 6 # reward fire action close sight
             
            elif fire_action == 0 :
                tank.reward += 2 # rencourage fire action
            

            if tank.number_of_ammo == 0:
                tank.reward -= 60  # Penalty for running out of ammo
        
        tank.reward -= 3 # force them to do something

        tank.total_reward += tank.reward
        return self.get_state(num_tank), tank.reward, done, {}

    def lost(self, tank):
        return tank.health <= 0



    def draw_hitboxes(self, draw = False):
        if draw :
            # Draw hitboxes around the tanks
            pygame.draw.rect(self.screen, (255, 0, 0), self.tank_1.rect, 2)  # Red hitbox for tank 1
            pygame.draw.rect(self.screen, (0, 255, 0), self.tank_2.rect, 2)  # Green hitbox for tank 2

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
                color = (255, 0, 0)
            else :
                color = (0, 255, 0) 
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
    
    def draw_text(self, tank, num_tank, epsilon = None):
        score_font = pygame.font.Font(pygame.font.get_default_font(), 14)
        if num_tank == 1 :
            score_text = score_font.render(f"[Tank 1] Current reward : {tank.total_reward}", True, (255, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.topleft = (10, 10)
        else :
            score_text = score_font.render(f"[Tank 2] Current reward : {tank.total_reward}", True, (0, 255, 0))
            score_rect = score_text.get_rect()
            score_rect.topright = (SCREEN_WIDTH -10, 10)
        self.screen.blit(score_text, score_rect)

        if epsilon is not None :
            score_text = score_font.render(f"Randomness : {epsilon*100:.1f}%", True, (0, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.topleft = (SCREEN_WIDTH // 2-50, 10)
        
        self.screen.blit(score_text, score_rect)



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
            print(game.get_state(num_tank = 1))

        # Check for bullet hits
        game.check_bullet_collisions()

        # Render everything
        game.render(rendering=True, clock=100)

        pygame.display.flip()




# tanks_train.py
import numpy as np
import random as rd
import multiprocessing as mp

import pygame
import torch
import time
import matplotlib.pyplot as plt

# Device selection
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Hyperparameters
EPISODES = 10_000
GAMMA = 0.999
ALPHA = 0.005
GLOBAL_N = 11
MAX_STEPS = 500 
EPS_DECAY = 0.95
STATE_SIZE = 23

# Function to run a single episode in parallel
def run_episode(agent_1, agent_2, epsilon, rendering, episode = 0):
    # Create a new environment inside the process
    env = TanksGame()

    print(f'---Running episode {episode} ---')
    env.reset()
    done = False
    total_reward_1 = 0
    total_reward_2 = 0
    steps = 0

    while not done and steps < MAX_STEPS:
        state_1 = env.get_state(num_tank=1)
        actions_1 = agent_1.get_actions(state_1, epsilon)
        next_state_1, reward_1, done, _ = env.step(actions_1, num_tank=1)
        agent_1.remember(state_1, actions_1, reward_1, next_state_1, done)
        agent_1.train_model(state_1, actions_1, reward_1, next_state_1, done)
        total_reward_1 += reward_1
        
        state_2 = env.get_state(num_tank=2)
        actions_2 = agent_2.get_actions(state_2, epsilon)
        next_state_2, reward_2, done, _ = env.step(actions_2, num_tank=2)
        agent_2.remember(state_2, actions_2, reward_2, next_state_2, done)
        agent_2.train_model(state_2, actions_2, reward_2, next_state_2, done)
        total_reward_2 += reward_2

        env.render(rendering = rendering, clock = 100, epsilon = epsilon)

        steps += 1
    
    return total_reward_1, total_reward_2, steps

# Parallel execution
def parallel_train(agent_1, agent_2, num_episodes=10, num_processes=4, episode = 0):
    pool = mp.Pool(processes=num_processes)
    results = []

    for _ in range(num_episodes):
        epsilon = max(0.01, EPS_DECAY ** episode)
        results.append(pool.apply_async(run_episode, args=(agent_1, agent_2, epsilon, False, episode)))

    # Close the pool and wait for the processes to complete
    pool.close()
    pool.join()

    # Gather results
    rewards_1 = [result.get()[0] for result in results]
    rewards_2 = [result.get()[1] for result in results]
    steps = [result.get()[2] for result in results]

    print(f"Average Reward Agent 1: {np.mean(rewards_1)}")
    print(f"Average Reward Agent 2: {np.mean(rewards_2)}")
    print(f"Average Steps: {np.mean(steps)}")

# Main Training Loop
def main_training_loop(agent_1, agent_2, EPISODES, render_every, rendering):
    for episode in range(EPISODES):
        epsilon = max(0.01, EPS_DECAY ** episode)

        if episode % render_every == 0:
            # Render this episode for visualization
            total_reward_1, total_reward_2, steps = run_episode(agent_1, agent_2, epsilon, rendering, episode)
            
            agent_1.replay()
            agent_2.replay()

            print(f'Episode: {episode + 1}, Total Reward Agent 1: {total_reward_1}, Total Reward Agent 2: {total_reward_2}, Steps: {steps}')

        else:
            # Run episodes in parallel
            parallel_train(agent_1, agent_2, num_episodes = 20, num_processes=4, episode = episode)
            
        # Save the trained models every 50 episodes
        if episode % 10 == 9:
            torch.save(agent_1.model.state_dict(), TANK_1_WEIGHTS + f"_epoch_{episode+1}.pth")
            torch.save(agent_2.model.state_dict(), TANK_2_WEIGHTS + f"_epoch_{episode+1}.pth")


if __name__ == "__main__":
    # Create the Q-learning agent
    agent_1 = TanksAgent(
        state_size=STATE_SIZE,
        action_sizes=[3, 3, 3, 2], # [move, rotate, strafe, fire]
        gamma = GAMMA,
        learning_rate = ALPHA,
        device = device,
        load_model = True,
    )

    agent_2 = TanksAgent(
        state_size=STATE_SIZE,
        action_sizes=[3, 3, 3, 2], # [move, rotate, strafe, fire]
        gamma = GAMMA,
        learning_rate = ALPHA,
        device = device,
        load_model = True,
    )

    if agent_1.load_model:
        print("Loading model 1 weights...")
        agent_1.model.load_state_dict(torch.load(TANK_1_WEIGHTS, map_location=device))

    if agent_2.load_model:
        print("Loading model 2 weights...")
        agent_2.model.load_state_dict(torch.load(TANK_2_WEIGHTS, map_location=device))

    # Start the training loop
    main_training_loop(agent_1, agent_2, EPISODES=10, render_every=2, rendering = True)


