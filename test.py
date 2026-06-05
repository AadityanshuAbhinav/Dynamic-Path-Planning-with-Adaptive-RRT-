import pygame
import random
import numpy as np
from shapely.geometry import Polygon, LineString, Point

# Constants
WIDTH, HEIGHT = 700, 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ROBOT_RADIUS = 10
STEP_SIZE = 15
MAX_ITER = 1500
GOAL_SAMPLE_RATE = 0.05
NEIGHBOR_RADIUS = 40

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (200, 200, 200)


pygame.init()
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RRT* with Moving Obstacles")
clock = pygame.time.Clock()

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class MovingObstacle:
    def __init__(self, points, velocity):
        self.original_points = np.array(points, dtype=np.float64)
        self.velocity = np.array(velocity)
        self.polygon = Polygon(self.original_points)

    def update(self):
        self.original_points += self.velocity
        self.polygon = Polygon(self.original_points)
        # Bounce off the screen edges
        if np.any(self.original_points[:, 0] <= 0) or np.any(self.original_points[:, 0] >= WIDTH):
            self.velocity[0] *= -1
        if np.any(self.original_points[:, 1] <= 0) or np.any(self.original_points[:, 1] >= HEIGHT):
            self.velocity[1] *= -1

    def draw(self, surface):
        pygame.draw.polygon(surface, (200, 50, 50), self.polygon.exterior.coords)

class SquareMover(MovingObstacle):
    def __init__(self, start_pos, size, travel_distance, speed, pause_duration):
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.size = size
        self.travel_distance = travel_distance
        self.speed = speed
        self.pause_duration = pause_duration * 1000  # Convert to milliseconds
        self.state = "moving_down"  # can be 'moving_down', 'pause_down', 'moving_up', 'pause_up'
        self.last_switch_time = pygame.time.get_ticks()

        points = [
            self.start_pos,
            self.start_pos + [size, 0],
            self.start_pos + [size, size],
            self.start_pos + [0, size]
        ]
        super().__init__(points, [0, speed])

    def update(self):
        current_time = pygame.time.get_ticks()
        if self.state == "moving_down":
            if self.original_points[0][1] >= self.start_pos[1] + self.travel_distance:
                self.velocity[1] = 0
                self.state = "pause_down"
                self.last_switch_time = current_time
            else:
                self.velocity[1] = self.speed

        elif self.state == "pause_down":
            if current_time - self.last_switch_time >= self.pause_duration:
                self.velocity[1] = -self.speed
                self.state = "moving_up"

        elif self.state == "moving_up":
            if self.original_points[0][1] <= self.start_pos[1]:
                self.velocity[1] = 0
                self.state = "pause_up"
                self.last_switch_time = current_time
            else:
                self.velocity[1] = -self.speed

        elif self.state == "pause_up":
            if current_time - self.last_switch_time >= self.pause_duration:
                self.velocity[1] = self.speed
                self.state = "moving_down"

        self.original_points += self.velocity
        self.polygon = Polygon(self.original_points)


class RRTStar:
    def __init__(self, start, goal, obstacles):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.nodes = [self.start]
        self.path = None
        self.obstacles = obstacles

    def get_random_node(self):
        if random.random() < GOAL_SAMPLE_RATE:
            return self.goal
        return Node(random.randint(0, WIDTH), random.randint(0, HEIGHT))

    def get_nearest_node(self, node):
        return min(self.nodes, key=lambda n: (n.x - node.x)**2 + (n.y - node.y)**2)

    def steer(self, from_node, to_node):
        dx, dy = to_node.x - from_node.x, to_node.y - from_node.y
        dist = np.hypot(dx, dy)
        if dist == 0:
            return from_node
        scale = min(STEP_SIZE / dist, 1.0)
        return Node(from_node.x + dx * scale, from_node.y + dy * scale)

    def check_collision(self, start, end):
        line = LineString([start, end])
        expanded_line = line.buffer(ROBOT_RADIUS)
        for obs in self.obstacles:
            if expanded_line.intersects(obs):
                return True
        for mob in moving_obstacles:
            if expanded_line.intersects(mob.polygon):
                return True
        return False

    def is_point_valid(self, point):
        p = Point(point)
        circle = p.buffer(ROBOT_RADIUS)
        for obs in self.obstacles:
            if circle.intersects(obs):
                return False
        for mob in moving_obstacles:
            if circle.intersects(mob.polygon):
                return False
        return True

    def is_path_obstructed(self):
        if not self.path:
            return None

        for i in range(len(self.path) - 1):
            start = self.path[i]
            end = self.path[i + 1]
            line = LineString([start, end]).buffer(ROBOT_RADIUS)

            for mob in moving_obstacles:
                if line.intersects(mob.polygon):
                    return i  # index of the obstructed segment

        return None


    def rewire(self, new_node, neighbors):
        for node in neighbors:
            potential_cost = new_node.cost + np.hypot(new_node.x - node.x, new_node.y - node.y)
            if potential_cost < node.cost and not self.check_collision((new_node.x, new_node.y), (node.x, node.y)):
                node.parent = new_node
                node.cost = potential_cost

    def find_path(self):
        for _ in range(MAX_ITER):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node)

            if not self.is_point_valid((new_node.x, new_node.y)):
                continue

            if self.check_collision((nearest_node.x, nearest_node.y), (new_node.x, new_node.y)):
                continue

            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + np.hypot(nearest_node.x - new_node.x, nearest_node.y - new_node.y)
            neighbors = [node for node in self.nodes if np.hypot(node.x - new_node.x, node.y - new_node.y) < NEIGHBOR_RADIUS]
            min_cost = new_node.cost
            min_parent = nearest_node

            for neighbor in neighbors:
                cost = neighbor.cost + np.hypot(neighbor.x - new_node.x, neighbor.y - new_node.y)
                if cost < min_cost and not self.check_collision((neighbor.x, neighbor.y), (new_node.x, new_node.y)):
                    min_cost = cost
                    min_parent = neighbor

            new_node.parent = min_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            self.rewire(new_node, neighbors)

            if np.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) < STEP_SIZE:
                if not self.check_collision((new_node.x, new_node.y), (self.goal.x, self.goal.y)):
                    self.goal.parent = new_node
                    self.path = self.extract_path()
                    return self.path

        return None

    def find_path_step(self):
        if len(self.nodes) >= MAX_ITER:
            return False

        rnd_node = self.get_random_node()
        nearest_node = self.get_nearest_node(rnd_node)
        new_node = self.steer(nearest_node, rnd_node)

        if not self.is_point_valid((new_node.x, new_node.y)):
            return False

        if self.check_collision((nearest_node.x, nearest_node.y), (new_node.x, new_node.y)):
            return False

        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + np.hypot(nearest_node.x - new_node.x, nearest_node.y - new_node.y)
        neighbors = [node for node in self.nodes if np.hypot(node.x - new_node.x, node.y - new_node.y) < NEIGHBOR_RADIUS]

        min_cost = new_node.cost
        min_parent = nearest_node

        for neighbor in neighbors:
            cost = neighbor.cost + np.hypot(neighbor.x - new_node.x, neighbor.y - new_node.y)
            if cost < min_cost and not self.check_collision((neighbor.x, neighbor.y), (new_node.x, new_node.y)):
                min_cost = cost
                min_parent = neighbor

        new_node.parent = min_parent
        new_node.cost = min_cost
        self.nodes.append(new_node)
        self.rewire(new_node, neighbors)

        if np.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) < STEP_SIZE:
            if not self.check_collision((new_node.x, new_node.y), (self.goal.x, self.goal.y)):
                self.goal.parent = new_node
                self.path = self.extract_path()
                return True

        return False

    def extract_path(self):
        path = []
        node = self.goal
        # print(node)
        while node is not None and node.parent is not None:
            path.append((node.x, node.y))
            node = node.parent
            # print(node.parent)
        if node is not None:
            path.append((self.start.x, self.start.y))
        path.reverse()
        return path

    def draw(self, surface):
        for node in self.nodes:
            if node.parent:
                pygame.draw.line(surface, (0, 0, 255), (node.x, node.y), (node.parent.x, node.parent.y), 1)

        if self.path:
            pygame.draw.lines(surface, (0, 255, 0), False, self.path, 3)

# Define static obstacles
obstacles = [
    Polygon([(100, 0), (400, 0), (400, 200), (100, 200)]),
    Polygon([(50, 300), (150, 300), (150, 600), (50, 600)]),
    Polygon([(300, 300), (500, 300), (500, 500), (300, 500)]),
    Polygon([(600, 300), (700, 300), (700, 400), (600, 400)]),
    Polygon([(200, 600), (500, 600), (500, 700), (200, 700)])
]

# Define moving obstacles
# moving_obstacles = [
#     MovingObstacle([(100, 300), (150, 300), (150, 350), (100, 350)], [1, 0.5]),
#     MovingObstacle([(500, 400), (550, 400), (550, 450), (500, 450)], [-1, -0.5])
# ]

# One square (100x100) that moves vertically between y = 200 and y = 300
moving_obstacles = [
    SquareMover(start_pos=(300, 100), size=100, travel_distance=100, speed=0.67, pause_duration=2)
]


start = (50, 50)
goal = (450, 250)

rrt = RRTStar(start, goal, obstacles)
rrt.find_path()
planning = False

running = True
while running:
    display.fill(WHITE)

    # Draw start and goal
    pygame.draw.circle(display, BLUE, start, 10)
    pygame.draw.circle(display, GREEN, goal, 10)

    # Update moving obstacles
    for mob in moving_obstacles:
        mob.update()

    # Draw static obstacles
    for obs in obstacles:
        pygame.draw.polygon(display, BLACK, list(obs.exterior.coords))

    # Draw moving obstacles
    for mob in moving_obstacles:
        mob.draw(display)

    # Draw RRT*
    rrt.draw(display)

    if not planning:
        obstructed_index = rrt.is_path_obstructed()
        if obstructed_index is not None:
            print(f"Obstacle encountered at segment {obstructed_index}")
            last_safe = rrt.path[max(0, obstructed_index - 1)]
            rrt = RRTStar(last_safe, goal, obstacles)
            planning = True
    else:
        success = rrt.find_path_step()
        if success:
            print("Replan complete!")
            planning = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
    clock.tick(60)

pygame.quit()
