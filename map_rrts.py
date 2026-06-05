import pygame
import numpy as np
import math
import sys
import random
from shapely.geometry import Polygon, LineString, Point

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 700, 700
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pioneer 3-AT RRT* Path Planning')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (200, 200, 200)

# Robot parameters (1px = 1cm in simulation)
ROBOT_RADIUS = 15  # 15cm radius for Pioneer 3-AT
MIN_TURN_RADIUS = 30  # 30cm minimum turning radius
STEP_SIZE = 35  # 35cm step size
SEARCH_RADIUS = 70  # 70cm search radius

# Define obstacles using Shapely Polygons
obstacles = [
    Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
    Polygon([(200, 100), (400, 100), (400, 200), (300, 200), (300, 300), (200, 300)]),
    Polygon([(500, 0), (600, 0), (600, 300), (500, 300)]),
    Polygon([(0, 300), (100, 300), (100, 500), (300, 500), (300, 600), (0, 600)]),
    Polygon([(400, 400), (500, 400), (500, 500), (600, 500), (600, 600), (400, 600)]),
    Polygon([(600, 500), (700, 500), (700, 700), (600, 700)])
]

class Node:
    def __init__(self, pos):
        self.pos = pos
        self.parent = None
        self.cost = 0.0

    def __iter__(self):
        return iter(self.pos)

class RRTStar:
    def __init__(self, start, goal, obstacles):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.nodes = [self.start]
        self.goal_threshold = 10
        self.max_iter = 5000
        self.path = None

    def distance(self, a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def check_collision(self, start, end):
        line = LineString([start, end])
        expanded_line = line.buffer(ROBOT_RADIUS)
        for obstacle in self.obstacles:
            if expanded_line.intersects(obstacle):
                return True
        return False

    def sample_point(self):
        if np.random.random() < 0.1:
            return self.goal.pos
        return (np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT))

    def find_nearest(self, point):
        return min(self.nodes, key=lambda node: self.distance(node.pos, point))

    def steer(self, from_node, to_point):
        vector = np.array(to_point) - np.array(from_node.pos)
        length = np.linalg.norm(vector)
        if length == 0:
            return from_node.pos
        direction = vector / length
        new_point = np.array(from_node.pos) + direction * min(STEP_SIZE, length)
        return tuple(new_point)

    def find_near_nodes(self, new_node):
        return [node for node in self.nodes 
                if self.distance(node.pos, new_node.pos) < SEARCH_RADIUS]

    def choose_parent(self, new_node, near_nodes):
        for node in near_nodes:
            if not self.check_collision(node.pos, new_node.pos):
                cost = node.cost + self.distance(node.pos, new_node.pos)
                if cost < new_node.cost:
                    new_node.parent = node
                    new_node.cost = cost

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            if not self.check_collision(new_node.pos, node.pos):
                cost = new_node.cost + self.distance(new_node.pos, node.pos)
                if cost < node.cost:
                    node.parent = new_node
                    node.cost = cost

    def find_path(self):
        for _ in range(self.max_iter):
            rand_point = self.sample_point()
            nearest = self.find_nearest(rand_point)
            new_point = self.steer(nearest, rand_point)
            
            if not self.check_collision(nearest.pos, new_point):
                new_node = Node(new_point)
                new_node.parent = nearest
                new_node.cost = nearest.cost + self.distance(nearest.pos, new_node.pos)
                
                near_nodes = self.find_near_nodes(new_node)
                self.choose_parent(new_node, near_nodes)
                
                self.nodes.append(new_node)
                self.rewire(new_node, near_nodes)
                
                if self.distance(new_node.pos, self.goal.pos) < self.goal_threshold:
                    self.path = self.extract_path(new_node)
                    return self.path
        return None

    def extract_path(self, node):
        path = []
        current = node
        while current.parent:
            path.append(current.pos)
            current = current.parent
        path.append(self.start.pos)
        return path[::-1]

    def smooth_path(self, path):
        if not path or len(path) < 3:
            return path
            
        smoothed = [path[0]]
        for i in range(1, len(path)-1):
            p0 = np.array(smoothed[-1])
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            
            # Cubic Bézier interpolation
            t = np.linspace(0, 1, 10)
            points = [(1-t)**2 * p0 + 2*(1-t)*t*p1 + t**2*p2 for t in t]
            
            valid = True
            for p in points:
                if self.check_collision(smoothed[-1], tuple(p)):
                    valid = False
                    break
            if valid:
                smoothed.append(path[i])
        smoothed.append(path[-1])
        return smoothed

def draw_obstacles(surface):
    for obstacle in obstacles:
        pygame.draw.polygon(surface, BLACK, list(obstacle.exterior.coords), 0)

def is_point_valid(point, obstacles):
    """Check if a point is in free space considering robot radius"""
    robot_area = Point(point).buffer(ROBOT_RADIUS)
    for obstacle in obstacles:
        if robot_area.intersects(obstacle):
            return False
    return True

def generate_valid_point(obstacles):
    """Generate random points until finding one in free space"""
    while True:
        x = random.randint(ROBOT_RADIUS, WIDTH-ROBOT_RADIUS)
        y = random.randint(ROBOT_RADIUS, HEIGHT-ROBOT_RADIUS)
        if is_point_valid((x, y), obstacles):
            return (x, y)

def main():
    # Generate valid random start and goal
    # start = generate_valid_point(obstacles)
    # goal = generate_valid_point(obstacles)

    # User defined start and goal
    start = (630, 50)
    goal = (420, 20)

    # Ensure start and goal are not too close
    while np.hypot(start[0]-goal[0], start[1]-goal[1]) < 200:
        goal = generate_valid_point(obstacles)
    
    rrt = RRTStar(start, goal, obstacles)
    running = True
    path_found = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not path_found:
            path = rrt.find_path()
            if path:
                smoothed_path = rrt.smooth_path(path)
                path_found = True

        display.fill(WHITE)
        draw_obstacles(display)
        
        # Draw tree
        for node in rrt.nodes:
            if node.parent:
                pygame.draw.line(display, GREY, node.pos, node.parent.pos, 1)
        
        # Draw start and goal
        pygame.draw.circle(display, BLUE, start, 10)
        pygame.draw.circle(display, GREEN, goal, 10)
        
        # Draw final path
        if path_found and smoothed_path:
            pygame.draw.lines(display, RED, False, smoothed_path, 3)
        
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
