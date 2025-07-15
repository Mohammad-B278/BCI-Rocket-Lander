import pygame
from settings import *

class Rocket:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.vx = 0.0
        self.vy = 0.0

        self.image = pygame.image.load('assets/images/rocket.png') 
        self.rect = self.image.get_rect() 
        self.rect.center = (self.x, self.y) 

        self.gravity = GRAVITY # e.g., 0.05
        self.thrust = THRUST   # e.g., -0.1
        self.side_thrust = SIDE_THRUST # e.g., 0.08
        self.fuel = 100
    
    def update(self, keys):
        self.vy += self.gravity

        if keys[pygame.K_UP] and self.fuel > 0:
            self.vy += self.thrust # Thrust is negative, so this pushes us up
            self.fuel -= 1 # Use some fuel

        if keys[pygame.K_LEFT] and self.fuel > 0:
            self.vx -= self.side_thrust
            self.fuel -= 0.5 # Side thrusters might use less fuel

        if keys[pygame.K_RIGHT] and self.fuel > 0:
            self.vx += self.side_thrust
            self.fuel -= 0.5

        self.x += self.vx
        self.y += self.vy

        self.rect.center = (self.x, self.y)

    def draw(self, surface):
        surface.blit(self.image, self.rect)

