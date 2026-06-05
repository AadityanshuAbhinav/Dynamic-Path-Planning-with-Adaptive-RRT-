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
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

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

class CircularMover:
    def __init__(self, center, path_size, radius, speed, pause_duration):
        self.center = np.array(center, dtype=np.float64)
        self.path_size = path_size
        self.radius = radius
        self.speed = speed
        self.pause_duration = pause_duration * 1000  # Convert to milliseconds
        self.state = "moving_right"
        self.last_switch_time = pygame.time.get_ticks()

        self.position = np.array([self.center[0] - path_size / 2, self.center[1] - path_size / 2], dtype=np.float64)

    def update(self):
        current_time = pygame.time.get_ticks()

        if "pause" in self.state:
            if current_time - self.last_switch_time >= self.pause_duration:
                if self.state == "pause_right":
                    self.state = "moving_down"
                elif self.state == "pause_down":
                    self.state = "moving_left"
                elif self.state == "pause_left":
                    self.state = "moving_up"
                elif self.state == "pause_up":
                    self.state = "moving_right"
            else:
                return  # During pause, don't move

        if self.state == "moving_right":
            self.position[0] += self.speed
            if self.position[0] >= self.center[0] + self.path_size / 2:
                self.position[0] = self.center[0] + self.path_size / 2
                self.state = "pause_right"
                self.last_switch_time = current_time

        elif self.state == "moving_down":
            self.position[1] += self.speed
            if self.position[1] >= self.center[1] + self.path_size / 2:
                self.position[1] = self.center[1] + self.path_size / 2
                self.state = "pause_down"
                self.last_switch_time = current_time

        elif self.state == "moving_left":
            self.position[0] -= self.speed
            if self.position[0] <= self.center[0] - self.path_size / 2:
                self.position[0] = self.center[0] - self.path_size / 2
                self.state = "pause_left"
                self.last_switch_time = current_time

        elif self.state == "moving_up":
            self.position[1] -= self.speed
            if self.position[1] <= self.center[1] - self.path_size / 2:
                self.position[1] = self.center[1] - self.path_size / 2
                self.state = "pause_up"
                self.last_switch_time = current_time

    def draw(self, surface):
        pygame.draw.circle(surface, (150, 0, 0), self.position.astype(int), self.radius)

    @property
    def polygon(self):
        # Approximate circle as a polygon for collision detection
        return Point(self.position).buffer(self.radius)


class MovingObstacle:
    def __init__(self, start_pos, radius, speed):
        self.position = np.array(start_pos, dtype=np.float64)
        self.target_position = np.array(start_pos, dtype=np.float64)
        self.radius = radius
        self.speed = speed

    def set_target(self, target_pos):
        self.target_position = np.array(target_pos, dtype=np.float64)

    def update(self):
        direction = self.target_position - self.position
        distance = np.linalg.norm(direction)
        if distance < self.speed:
            self.position = self.target_position  # Snap to target
        elif distance > 0:
            direction /= distance
            move = direction * self.speed
            self.position += move

    def draw(self, surface):
        pygame.draw.circle(surface, (200, 50, 50), self.position.astype(int), self.radius)

    @property
    def polygon(self):
        return Point(self.position).buffer(self.radius)


class SquareMover(MovingObstacle):
    def __init__(self, start_pos, size, travel_distance, speed, pause_duration):
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.size = size
        self.travel_distance = travel_distance
        self.speed = speed
        self.pause_duration = pause_duration * 1000
        self.state = "moving_down"
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
    def __init__(self, start, goal, obstacles, temporary_obstacles=[]):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.nodes = [self.start]
        self.path = None
        self.obstacles = obstacles
        self.temporary_obstacles = temporary_obstacles

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
        for obs in self.obstacles + self.temporary_obstacles:
            if expanded_line.intersects(obs):
                return True
        return False

    def is_point_valid(self, point):
        p = Point(point)
        circle = p.buffer(ROBOT_RADIUS)
        for obs in self.obstacles + self.temporary_obstacles:
            if circle.intersects(obs):
                return False
        return True

    def is_path_obstructed(self, current_robot_index=0, lookahead=50):
        if not self.path:
            return None
        path_to_check = self.path[current_robot_index : current_robot_index + lookahead]
        for i in range(len(path_to_check) - 1):
            start = path_to_check[i]
            end = path_to_check[i + 1]
            line = LineString([start, end]).buffer(ROBOT_RADIUS)
            for mob in moving_obstacles:
                if line.intersects(mob.polygon):
                    return current_robot_index + i
        return None


    def rewire(self, new_node, neighbors):
        for node in neighbors:
            potential_cost = new_node.cost + np.hypot(new_node.x - node.x, new_node.y - node.y)
            if potential_cost < node.cost and not self.check_collision((new_node.x, new_node.y), (node.x, node.y)):
                node.parent = new_node
                node.cost = potential_cost

    def find_path(self):
        for _ in range(MAX_ITER):
            if self.find_path_step():
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
        while node and node.parent:
            path.append((node.x, node.y))
            node = node.parent
        if node:
            path.append((self.start.x, self.start.y))
        path.reverse()
        return path

    def draw(self, surface):
        if self.path:
            pygame.draw.lines(surface, (0, 255, 0), False, self.path, 3)

class Robot:
    def __init__(self, path, speed):
        self.path = path
        self.speed = speed
        self.current_index = 0
        self.position = np.array(self.path[0], dtype=np.float64)

    def update(self):
        if self.current_index >= len(self.path) - 1:
            return  # already at goal

        start = np.array(self.path[self.current_index], dtype=np.float64)
        end = np.array(self.path[self.current_index + 1], dtype=np.float64)
        direction = end - start
        distance = np.linalg.norm(direction)

        if distance == 0:
            self.current_index += 1
            return

        direction /= distance  # normalize
        move = direction * self.speed

        if np.linalg.norm(end - self.position) <= self.speed:
            self.position = end
            self.current_index += 1
        else:
            self.position += move

    def draw(self, surface):
        pygame.draw.circle(surface, (0, 0, 255), self.position.astype(int), ROBOT_RADIUS)

    def reached_goal(self):
        return self.current_index >= len(self.path) - 1

# Static obstacles
obstacles = [
    Polygon([(100, 0), (400, 0), (400, 200), (100, 200)]),
    Polygon([(50, 300), (150, 300), (150, 600), (50, 600)]),
    Polygon([(300, 300), (500, 300), (500, 500), (300, 500)]),
    Polygon([(600, 300), (700, 300), (700, 400), (600, 400)]),
    Polygon([(200, 600), (500, 600), (500, 700), (200, 700)])
]

moving_obstacles = [
    MovingObstacle(start_pos=(WIDTH//2, HEIGHT//2), radius=20, speed=2)
]

# Set the target for obstacle
moving_obstacles[0].set_target((WIDTH//2 + 100, HEIGHT//2 + 100))  # Example new target

temporary_obstacles = []
start = (50, 50)
goal = (650, 450)

rrt = RRTStar(start, goal, obstacles)
rrt.find_path()
robot = Robot(rrt.path, speed=2)  # speed is in pixels per frame
planning = False
old_paths = []

running = True
while running:
    display.fill(WHITE)
    pygame.draw.circle(display, BLUE, start, 10)
    pygame.draw.circle(display, GREEN, goal, 10)

    for mob in moving_obstacles:
        mob.update()

    for obs in obstacles:
        pygame.draw.polygon(display, BLACK, list(obs.exterior.coords))

    for mob in moving_obstacles:
        mob.draw(display)

    for path in old_paths:
        if len(path) >= 2:
            pygame.draw.lines(display, (0, 200, 0), False, path, 3)
    
    # Move the robot
    if not planning and robot.path:
        robot.update()

    # Draw robot
    if robot.path:
        robot.draw(display)

    rrt.draw(display)

    if not planning and not robot.reached_goal():
        obstructed_index = rrt.is_path_obstructed(robot.current_index)
        if obstructed_index is not None:
            print(f"Obstacle encountered at future segment {obstructed_index}")

            safe_path = rrt.path[:obstructed_index + 1]
            old_paths.append(safe_path)

            # Take snapshot of moving obstacles
            temporary_obstacles = [Polygon(mob.polygon.exterior.coords) for mob in moving_obstacles]

            new_start = tuple(robot.position)

            rrt = RRTStar(new_start, goal, obstacles, temporary_obstacles)
            planning = True

    elif planning:
        success = rrt.find_path_step()
        if success:
            print("Replan complete!")
            planning = False
            temporary_obstacles = []

            # Build new path starting from robot's actual current position
            new_path = [(robot.position[0], robot.position[1])] + rrt.path
            robot = Robot(new_path, speed=2)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
    clock.tick(60)

pygame.quit()
