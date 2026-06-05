import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH, SCREEN_HEIGHT = 700, 700
GRID_SIZE = 35  # Each grid cell size
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Warehouse Map with Pioneer 3-AT")

# Colors
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
DARK_GREY = (100, 100, 100)
YELLOW = (255, 215, 0)
BROWN = (139, 69, 19)
BLUE = (100, 149, 237)
BLACK = (0, 0, 0)

# Pioneer 3-AT robot settings
robot_radius = 15
robot_pos = [SCREEN_WIDTH//2, SCREEN_HEIGHT//2]

# Objects (approximate locations based on grid)
objects = [
    # (Color, Rect(x, y, width, height))
    (YELLOW, pygame.Rect(70, 14, 308, 84)),   # Top shelf
    (BROWN, pygame.Rect(14, 140, 84, 133)),     # Table
    (BROWN, pygame.Rect(14, 283.5, 168, 168)),   # Left stack of boxes
    (BROWN, pygame.Rect(353.5, 292.25, 168, 168)),  # Middle messy boxes
    (BLUE, pygame.Rect(14, 502.25, 84, 114.25)),      # Bottom left blue small table
    (BROWN, pygame.Rect(602, 152.25, 84, 133)),   # Right tall stack
    (BLUE, pygame.Rect(99.75, 616, 105, 70)),      # Bottom left blue bin
    (GREY, pygame.Rect(248.5, 616, 105, 70)),     # Bottom center grey bin
    (YELLOW, pygame.Rect(376, 575.75, 308, 110.25)),  # Bottom shelf
    (GREY, pygame.Rect(598, 384, 88, 148)),    # Small table with packages
]

# Main loop
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(WHITE)

    # Draw floor grid
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GREY, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GREY, (0, y), (SCREEN_WIDTH, y))

    # Draw boundary walls
    wall_thickness = 14
    pygame.draw.rect(screen, BLACK, (0, 0, SCREEN_WIDTH, wall_thickness))  # Top
    pygame.draw.rect(screen, BLACK, (0, 0, wall_thickness, SCREEN_HEIGHT)) # Left
    pygame.draw.rect(screen, BLACK, (0, SCREEN_HEIGHT-wall_thickness, SCREEN_WIDTH, wall_thickness))  # Bottom
    pygame.draw.rect(screen, BLACK, (SCREEN_WIDTH-wall_thickness, 0, wall_thickness, SCREEN_HEIGHT))  # Right

    # Draw objects
    for color, rect in objects:
        pygame.draw.rect(screen, color, rect)

    # Draw Pioneer 3-AT robot
    pygame.draw.circle(screen, DARK_GREY, robot_pos, robot_radius)

    pygame.display.update()
    clock.tick(60)
