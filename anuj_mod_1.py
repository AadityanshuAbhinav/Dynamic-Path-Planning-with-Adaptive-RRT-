import pygame
import random
from shapely.geometry import LineString, Polygon, Point
from shapely.affinity import translate

# ------------------- Setup -------------------
WIDTH, HEIGHT = 700, 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (200, 50, 50)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RRT* with Moving Obstacles")

# ------------------- Environment -------------------
static_obstacles = [
    pygame.Rect(100, 100, 300, 100),
    pygame.Rect(50, 300, 100, 300),
    pygame.Rect(300, 400, 200, 200),
    pygame.Rect(500, 500, 100, 100)
]

moving_obstacle = pygame.Rect(300, 200, 100, 100)
moving_poly = Polygon([
    (moving_obstacle.left, moving_obstacle.top),
    (moving_obstacle.right, moving_obstacle.top),
    (moving_obstacle.right, moving_obstacle.bottom),
    (moving_obstacle.left, moving_obstacle.bottom)
])

def get_all_obstacles():
    polys = []
    for obs in static_obstacles:
        polys.append(Polygon([
            (obs.left, obs.top),
            (obs.right, obs.top),
            (obs.right, obs.bottom),
            (obs.left, obs.bottom)
        ]))
    polys.append(moving_poly)
    return polys

start = (50, 50)
goal = (600, 300)

# ------------------- RRT* Node -------------------
class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent
        self.cost = 0

# ------------------- RRT* Planner -------------------
class RRTStar:
    def __init__(self, start, goal, obstacles):
        self.start = Node(start)
        self.goal = Node(goal)
        self.nodes = [self.start]
        self.obstacles = obstacles
        self.goal_radius = 20
        self.step_size = 30
        self.search_radius = 50

    def collision(self, p1, p2):
        line = LineString([p1, p2])
        for obs in self.obstacles:
            if line.intersects(obs):
                return True
        return False

    def get_path(self):
        for _ in range(1000):
            rnd = (random.randint(0, WIDTH), random.randint(0, HEIGHT))
            nearest = min(self.nodes, key=lambda n: Point(n.pos).distance(Point(rnd)))
            dx = rnd[0] - nearest.pos[0]
            dy = rnd[1] - nearest.pos[1]
            mag = (dx**2 + dy**2)**0.5
            if mag == 0:
                continue  # Skip degenerate case
            new_pos = (
                nearest.pos[0] + self.step_size * (dx / mag),
                nearest.pos[1] + self.step_size * (dy / mag)
            )
            if not self.collision(nearest.pos, new_pos):
                new_node = Node(new_pos, nearest)
                new_node.cost = nearest.cost + Point(new_pos).distance(Point(nearest.pos))
                self.nodes.append(new_node)
                if Point(new_pos).distance(Point(self.goal.pos)) < self.goal_radius:
                    self.goal.parent = new_node
                    return self.extract_path(self.goal)
        return None

    def extract_path(self, node):
        path = []
        while node:
            path.append(node.pos)
            node = node.parent
        return path[::-1]

# ------------------- Robot State -------------------
robot_path = []
full_path = []
path_index = 0
current_pos = start

# ------------------- Main Loop -------------------
clock = pygame.time.Clock()
running = True
frame = 0

# Initial planning
planner = RRTStar(start, goal, get_all_obstacles())
robot_path = planner.get_path()
if robot_path:
    full_path = robot_path.copy()
else:
    full_path = []

while running:
    screen.fill(WHITE)

    # Draw static and moving obstacles
    for obs in static_obstacles:
        pygame.draw.rect(screen, BLACK, obs)
    pygame.draw.rect(screen, RED, moving_obstacle)

    # Update moving obstacle
    moving_obstacle.y += 1
    if moving_obstacle.y > 500:
        moving_obstacle.y = 200
    moving_poly = Polygon([
        (moving_obstacle.left, moving_obstacle.top),
        (moving_obstacle.right, moving_obstacle.top),
        (moving_obstacle.right, moving_obstacle.bottom),
        (moving_obstacle.left, moving_obstacle.bottom)
    ])

    # Robot reached goal
    if path_index >= len(full_path):
        pygame.draw.circle(screen, GREEN, goal, 8)
        pygame.display.flip()
        clock.tick(60)
        continue

    # Check if future path is blocked
    obstacles = get_all_obstacles()
    segment_blocked = False
    if path_index + 1 < len(full_path):
        p1, p2 = full_path[path_index], full_path[path_index + 1]
        line = LineString([p1, p2])
        for obs in obstacles:
            if line.intersects(obs):
                segment_blocked = True
                break

    # Replan if needed
    if segment_blocked:
        followed_path = full_path[:path_index + 1]
        planner = RRTStar(full_path[path_index], goal, obstacles)
        new_path = planner.get_path()
        if new_path:
            full_path = followed_path + new_path[1:]
            path_index = len(followed_path) - 1

    # Move along path
    if path_index < len(full_path):
        current_pos = full_path[path_index]
        path_index += 1

    # Draw full path
    if len(full_path) >= 2:
        pygame.draw.lines(screen, BLUE, False, full_path, 2)

    # Draw current robot position and goal
    pygame.draw.circle(screen, BLUE, start, 7)
    pygame.draw.circle(screen, GREEN, goal, 8)
    pygame.draw.circle(screen, (0, 255, 255), current_pos, 6)

    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
