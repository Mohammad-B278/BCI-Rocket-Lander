import pygame
from settings import *

class LandingPad:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.width = PAD_WIDTH
        self.height =PAD_HEIGHT

        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
    
    def draw(self, surface):
        pygame.draw.rect(surface, PAD_COLOUR, self.rect)

