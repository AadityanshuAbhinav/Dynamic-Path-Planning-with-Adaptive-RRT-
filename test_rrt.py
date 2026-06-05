import pygame
import random
import math
import numpy as np
from shapely.geometry import Polygon, Point

# Initialize pygame
pygame.init()

# Set display
WIDTH, HEIGHT = 800, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RRT with Moving Obstacle Prediction")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (150, 150, 150)

# Map Obstacles
obstacles = [
    Polygon([(100, 0), (400, 0), (400, 200), (100, 200)]),
    Polygon([(50, 300), (150, 300), (150, 600), (50, 600)]),
    Polygon([(300, 300), (500, 300), (500, 500), (300, 500)]),
    Polygon([(600, 300), (700, 300), (700, 400), (600, 400)]),
    Polygon([(200, 600), (500, 600), (500, 700), (200, 700)])
]

# RRT parameters
STEP_SIZE = 10
GOAL_SAMPLE_RATE = 0.1
MAX_ITER = 5000

# Robot parameters
robot_radius = 5
robot_speed = 2

# Prediction parameters
PREDICTION_HORIZON = 30
TEMP_OBSTACLE_BUFFER = 10

class Node:
    def __init__(self, pos):
        self.pos = pos
        self.parent = None

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def collision_check(p1, p2, obstacle_list):
    line = Point(p1).buffer(robot_radius).union(Point(p2).buffer(robot_radius)).convex_hull
    for obs in obstacle_list:
        if line.intersects(obs):
            return True
    return False

def get_random_point(goal):
    if random.random() < GOAL_SAMPLE_RATE:
        return goal
    else:
        return (random.randint(0, WIDTH), random.randint(0, HEIGHT))

def nearest(nodes, point):
    return min(nodes, key=lambda node: distance(node.pos, point))

def steer(from_node, to_point, step_size):
    theta = math.atan2(to_point[1] - from_node.pos[1], to_point[0] - from_node.pos[0])
    new_pos = (from_node.pos[0] + step_size * math.cos(theta), from_node.pos[1] + step_size * math.sin(theta))
    new_node = Node(new_pos)
    new_node.parent = from_node
    return new_node

def extract_path(last_node):
    path = []
    node = last_node
    while node is not None:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

def rrt(start, goal, obstacle_list, max_iter=MAX_ITER):
    nodes = [Node(start)]
    for _ in range(max_iter):
        rand_point = get_random_point(goal)
        nearest_node = nearest(nodes, rand_point)
        new_node = steer(nearest_node, rand_point, STEP_SIZE)

        if not collision_check(nearest_node.pos, new_node.pos, obstacle_list):
            nodes.append(new_node)

            if distance(new_node.pos, goal) < STEP_SIZE:
                final_node = steer(new_node, goal, STEP_SIZE)
                if not collision_check(new_node.pos, final_node.pos, obstacle_list):
                    final_node.parent = new_node
                    return extract_path(final_node)
    return None

def predict_collision(path, obstacle_pos, obstacle_dir, horizon, obstacle_speed):
    temp_pos = np.array(obstacle_pos, dtype=float)
    for idx, point in enumerate(path):
        temp_pos += obstacle_dir * obstacle_speed
        if distance(point, temp_pos) < robot_radius + 15:
            return idx
    return -1

def move_along_path(robot_pos, path, speed):
    if len(path) == 0:
        return robot_pos, []
    target = path[0]
    dist = distance(robot_pos, target)
    if dist < speed:
        return target, path[1:]
    else:
        theta = math.atan2(target[1] - robot_pos[1], target[0] - robot_pos[0])
        new_pos = (robot_pos[0] + speed * math.cos(theta), robot_pos[1] + speed * math.sin(theta))
        return new_pos, path

# Main function
def main():
    clock = pygame.time.Clock()

    start = (50, 750)
    goal = (750, 50)

    robot_pos = start

    # Moving obstacle
    obstacle_pos = np.array([400.0, 400.0])
    square_size = 200
    square_waypoints = [
        np.array([500, 400]),
        np.array([500, 600]),
        np.array([300, 600]),
        np.array([300, 400])
    ]
    current_waypoint = 0
    obstacle_speed = 2

    obstacle_radius = 20

    robot_path = rrt(start, goal, obstacles)
    if robot_path is None:
        print("Initial planning failed!")
        return

    temp_obstacles = []

    running = True
    while running:
        clock.tick(60)
        WIN.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw static obstacles
        for obs in obstacles:
            pygame.draw.polygon(WIN, GREY, list(obs.exterior.coords))

        # Draw temp obstacles
        for temp_obs in temp_obstacles:
            pygame.draw.circle(WIN, BLACK, (int(temp_obs.centroid.x), int(temp_obs.centroid.y)), obstacle_radius + TEMP_OBSTACLE_BUFFER, 2)

        # Draw goal and robot
        pygame.draw.circle(WIN, GREEN, (int(goal[0]), int(goal[1])), 8)
        pygame.draw.circle(WIN, RED, (int(robot_pos[0]), int(robot_pos[1])), 8)

        # Draw obstacle
        pygame.draw.circle(WIN, BLACK, (int(obstacle_pos[0]), int(obstacle_pos[1])), obstacle_radius)

        # Draw path
        if robot_path:
            for i in range(len(robot_path) - 1):
                pygame.draw.line(WIN, BLUE, robot_path[i], robot_path[i + 1], 2)

        # Robot move
        robot_pos, robot_path = move_along_path(robot_pos, robot_path, robot_speed)

        # Move obstacle along square
        waypoint = square_waypoints[current_waypoint]
        dir_vec = waypoint - obstacle_pos
        if np.linalg.norm(dir_vec) < 5:
            current_waypoint = (current_waypoint + 1) % len(square_waypoints)
            waypoint = square_waypoints[current_waypoint]
            dir_vec = waypoint - obstacle_pos
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        obstacle_pos += dir_vec * obstacle_speed

        # Predict collision
        predicted_idx = predict_collision(robot_path, obstacle_pos, dir_vec, PREDICTION_HORIZON, obstacle_speed)
        if predicted_idx != -1:
            print("Predicted collision at idx:", predicted_idx)
            # Insert temp obstacle
            danger_point = robot_path[predicted_idx]
            temp_circle = Point(danger_point).buffer(obstacle_radius + TEMP_OBSTACLE_BUFFER)
            obstacles.append(temp_circle)
            temp_obstacles.append(temp_circle)

            # Replan
            new_start_idx = max(predicted_idx - 5, 0)
            robot_pos = robot_path[new_start_idx]
            robot_path = rrt(robot_pos, goal, obstacles, max_iter=10000)

            if robot_path is None:
                print("Failed to replan!")
                running = False

        if distance(robot_pos, goal) < 10:
            print("Goal reached!")
            running = False

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
