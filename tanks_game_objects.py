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

