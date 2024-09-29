# tanks_game_objects.py
import pygame
import numpy as np
import random as rd
import time
from no_pygame_optimization import Rect
from tanks_paths import RENDERING

BULLET_COOLDOWN_TIME = 0.75
BULLET_SPEED = 15
TANK_HEALTH = 100
TANK_AMMO = 1000

class Background:
    def __init__(self, image_file, location, width, height, rendering):
        self.rendering = rendering
        if self.rendering:
            self.image = pygame.image.load(image_file)
            self.image = pygame.transform.scale(self.image, (width, height))
            self.rect = self.image.get_rect()
            self.rect.left, self.rect.top = location
        self.location = location
        self.width = width
        self.height = height

class TankPlayer:
    def __init__(self, image_file, location, width, speed, rendering):
        self.rendering = rendering
        self.location = np.array(location)
        self.width = width * 2  # Adjust width for the rect since the original image was scaled
        self.height = width
        if self.rendering:
            self.original_image = pygame.image.load(image_file).convert_alpha()
            self.original_image = pygame.transform.scale(self.original_image, (self.width, self.height))
            self.image = self.original_image.copy()
            self.rect = self.image.get_rect()
            self.rect.center = location
        else:
            self.rect = Rect(left=location[0] - self.width // 2, top=location[1] - self.height // 2, width=self.width, height=self.height)
        self.direction = 0
        self.speed = speed
        self.cooldown = time.time()
        self.bullets = pygame.sprite.Group() if self.rendering else []
        self.in_line_of_sight = False
        self.on_close_right = False
        self.on_close_left = False
        self.health = TANK_HEALTH
        self.number_of_ammo = TANK_AMMO
        self.reward = 0
        self.total_reward = 0
        self.was_hit = False
        
    def rotate(self, angle):
        self.direction += angle
        self.direction %= 360  # Keep the angle within 0-359 degrees
        if self.rendering:
            self.image = pygame.transform.rotate(self.original_image, self.direction)
            self.rect = self.image.get_rect(center=self.rect.center)
    
    def check_cooldown(self, current_time):
        return current_time - self.cooldown > BULLET_COOLDOWN_TIME
    
class Bullet(pygame.sprite.Sprite if RENDERING else object):
    def __init__(self, image_file, location, direction, width, screen_dims, rendering, game_instance, block=None):
        if rendering:
            super().__init__()
        self.rendering = rendering
        self.game_instance = game_instance  # Store reference to the game instance
        self.location = np.array(location)
        self.width = int(2.5 * width)
        self.height = width
        self.direction = direction
        self.speed = BULLET_SPEED
        self.screen_dims = screen_dims
        self.block = block  # Reference to the block in the arena
        if self.rendering:
            self.original_image = pygame.image.load(image_file).convert_alpha()
            self.original_image = pygame.transform.scale(self.original_image, (self.width, self.height))
            self.image = self.original_image
            self.rect = self.image.get_rect()
            self.rect.center = location
            self.image = pygame.transform.rotate(self.original_image, self.direction)
        else:
            self.rect = Rect(left=location[0] - self.width // 2, top=location[1] - self.height // 2, width=self.width, height=self.height)

    def update(self):
        rad_angle = np.radians(self.direction)
        self.location[0] += int(self.speed * np.cos(rad_angle))
        self.location[1] -= int(self.speed * np.sin(rad_angle))
        if self.rendering:
            self.rect.x, self.rect.y = self.location
        else:
            self.rect.topleft = (self.location[0] - self.width // 2, self.location[1] - self.height // 2)

    def check_collision(self, target_rect):
        return self.rect.colliderect(target_rect)
    
    def kill(self):
        if self.rendering:
            super().kill()  # Call the Pygame Sprite's kill method
        else:
            for tank in [self.game_instance.tank_1, self.game_instance.tank_2]:
                if self in tank.bullets:
                    tank.bullets.remove(self)  # Manually remove the bullet from the list

class Block:
    def __init__(self, image_file, location, width, height, rendering):
        self.rendering = rendering
        self.location = np.array(location)
        self.width = width
        self.height = height
        if self.rendering:
            self.image = pygame.image.load(image_file)
            self.image = pygame.transform.scale(self.image, (width, height))
            self.rect = self.image.get_rect()
            self.rect.center = location
        else:
            self.rect = Rect(left=location[0] - width // 2, top=location[1] - height // 2, width=width, height=height)


