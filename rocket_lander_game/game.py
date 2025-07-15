import pygame
import sys
from landing_pad import LandingPad
from rocket import Rocket
from settings import *

class Game:
    """Manages the main game loop and all game components."""
    def __init__(self):
        """Initialises the game window, objects, and settings."""
        pygame.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Rocket Lander")

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont('Arial', 24)

        self.landing_pad = LandingPad(x=350, y=550)
        self.rocket = Rocket(x=SCREEN_WIDTH / 2, y=50)

        self.game_state = "running" # Possible states: running, crashed, landed

    def main_loop(self):
        """Runs the main game loop, handling events, updates, and rendering."""
        while True: 
            self.handle_input()
            self.update_game_state()
            self.draw_elements()
            self.clock.tick(60)

    def handle_input(self):
        """Processes all user input from the keyboard and window events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()

        # Only allow player input if the game is active
        if self.game_state == "running":
            self.rocket.update(keys)

    def update_game_state(self):
        """Updates game logic and checks for win or loss conditions."""
        if self.game_state == "running":
            self.check_crash()
            self.check_landing_success()

    def draw_elements(self):
        """Draws all game objects and UI elements to the screen."""
        self.screen.fill(BLACK)

        self.rocket.draw(self.screen)
        self.landing_pad.draw(self.screen)

        self.draw_ui_text()

        if self.game_state == "crashed":
            self.display_message("CRASHED!", RED)
        elif self.game_state == "landed":
            self.display_message("LANDING SUCCESSFUL!", GREEN)

        pygame.display.flip()

    def draw_ui_text(self):
        """Renders and displays the user interface text (fuel, velocity)."""
        # Fuel text
        fuel_text = self.font.render(f"Fuel: {int(self.rocket.fuel)}", True, WHITE)
        self.screen.blit(fuel_text, (10, 10))

        # Velocity text
        vx_text = self.font.render(f"Vx: {self.rocket.vx:.2f}", True, WHITE)
        vy_text = self.font.render(f"Vy: {self.rocket.vy:.2f}", True, WHITE)
        self.screen.blit(vx_text, (10, 40))
        self.screen.blit(vy_text, (10, 70))
    
    def check_crash(self):
        """Checks if the rocket has crashed into the walls or ground."""
        if self.rocket.rect.left < 0 or self.rocket.rect.right > SCREEN_WIDTH:
            self.game_state = "crashed"
        
        # Crash if it hits the bottom of the screen and is not on the landing pad
        if self.rocket.rect.bottom > SCREEN_HEIGHT and not self.rocket.rect.colliderect(self.landing_pad.rect):
            self.game_state = "crashed"

    def check_landing_success(self):
        """Checks for a successful landing on the pad within safe velocity limits."""
        if self.rocket.rect.colliderect(self.landing_pad.rect):
            if self.rocket.vy < MAX_LANDING_SPEED and abs(self.rocket.vx) < MAX_LANDING_SPEED:
                self.game_state = "landed"
            else:
                self.game_state = "crashed" # Crashed due to high landing speed

    def display_message(self, text, color):
        """Displays a message in the centre of the screen."""
        message_surface = self.font.render(text, True, color)
        message_rect = message_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        self.screen.blit(message_surface, message_rect)