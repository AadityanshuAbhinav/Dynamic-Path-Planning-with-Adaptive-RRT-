import pygame
import random
import numpy as np
import time
from shapely.geometry import Polygon, LineString, Point

# ========== Constants ==========
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
clock = pygame.time.Clock()

# ========== Classes ==========

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
        self.pause_duration = pause_duration * 1000
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
                return

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

    @property
    def polygon(self):
        return Point(self.position).buffer(self.radius)

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
        line = LineString([start, end]).buffer(ROBOT_RADIUS)
        for obs in self.obstacles + self.temporary_obstacles:
            if line.intersects(obs):
                return True
        return False

    def is_point_valid(self, point):
        circle = Point(point).buffer(ROBOT_RADIUS)
        for obs in self.obstacles + self.temporary_obstacles:
            if circle.intersects(obs):
                return False
        return True

    def is_path_obstructed(self, current_robot_index=0):
        if not self.path:
            return None
        for i in range(current_robot_index, len(self.path) - 1):
            start = self.path[i]
            end = self.path[i + 1]
            line = LineString([start, end]).buffer(ROBOT_RADIUS)
            for mob in moving_obstacles:
                if line.intersects(mob.polygon):
                    return i
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

class Robot:
    def __init__(self, path, speed):
        self.path = path
        self.speed = speed
        self.current_index = 0
        self.position = np.array(self.path[0], dtype=np.float64)

    def update(self):
        if self.current_index >= len(self.path) - 1:
            return

        start = np.array(self.path[self.current_index], dtype=np.float64)
        end = np.array(self.path[self.current_index + 1], dtype=np.float64)
        direction = end - start
        distance = np.linalg.norm(direction)

        if distance == 0:
            self.current_index += 1
            return

        direction /= distance
        move = direction * self.speed

        if np.linalg.norm(end - self.position) <= self.speed:
            self.position = end
            self.current_index += 1
        else:
            self.position += move

    def reached_goal(self):
        return self.current_index >= len(self.path) - 1

# ========== Simulation and Metrics ==========

class MetricsTracker:
    def __init__(self):
        self.successful_runs = 0
        self.total_replans = 0
        self.total_replan_time = 0
        self.total_path_length = 0
        self.total_ideal_distance = 0
        self.near_collisions = 0
        self.metrics = []

    def record_run(self, success, replans, replan_time, path_length, ideal_distance, near_collision):
        if success:
            self.successful_runs += 1
        self.total_replans += replans
        self.total_replan_time += replan_time
        self.total_path_length += path_length
        self.total_ideal_distance += ideal_distance
        self.near_collisions += near_collision
        self.metrics.append({
            "Success": success,
            "Replans": replans,
            "Total Replan Time": replan_time,
            "Path Length": path_length,
            "Ideal Distance": ideal_distance,
            "Near Collisions": near_collisions
        })

    def save_to_csv(self, filename="simulation_metrics.csv"):
        if not self.metrics:
            print("No metrics to save.")
            return

        keys = self.metrics[0].keys()
        with open(filename, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.metrics)

        print(f"Metrics saved to {filename}")

    def report(self, runs):
        if runs == 0:
            print("No runs to report.")
            return

        success_rate = sum(1 for m in self.metrics if m["Success"]) / runs * 100
        avg_replans = sum(m["Replans"] for m in self.metrics) / runs
        avg_replan_time = sum(m["Total Replan Time"] for m in self.metrics) / runs
        avg_path_efficiency = sum(
            (m["Ideal Distance"] / m["Path Length"] if m["Path Length"] > 0 else 0)
            for m in self.metrics
        ) / runs
        avg_near_collisions = sum(m["Near Collisions"] for m in self.metrics) / runs

        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Average Number of Replans: {avg_replans:.2f}")
        print(f"Average Total Replan Time: {avg_replan_time:.2f} seconds")
        print(f"Average Path Efficiency: {avg_path_efficiency:.2f}")
        print(f"Average Near Collisions: {avg_near_collisions:.2f}")

    def report(self, total_runs):
        print("\n===== METRICS =====")
        print(f"Success rate: {self.successful_runs}/{total_runs} ({100*self.successful_runs/total_runs:.1f}%)")
        print(f"Avg replans per run: {self.total_replans/total_runs:.2f}")
        print(f"Avg replan time: {self.total_replan_time/self.total_replans if self.total_replans > 0 else 0:.2f} seconds")
        print(f"Avg path length ratio (actual/ideal): {self.total_path_length/self.total_ideal_distance:.2f}")
        print(f"Near collisions: {self.near_collisions}")

# ========== End of Setup ==========

def run_simulation():
    start = (50, 50)
    goal = (WIDTH - 50, HEIGHT - 50)
    obstacles = [
        Polygon([(200, 200), (300, 200), (300, 300), (200, 300)]),
        Polygon([(400, 400), (500, 400), (500, 500), (400, 500)]),
        Polygon([(100, 500), (150, 500), (150, 600), (100, 600)])
    ]

    global moving_obstacles
    moving_obstacles = [
        CircularMover(center=(WIDTH//2, HEIGHT//2), path_size=350, radius=20, speed=2, pause_duration=1)
    ]

    planner = RRTStar(start, goal, obstacles, temporary_obstacles=[mob.polygon for mob in moving_obstacles])
    path = planner.find_path()

    if path is None:
        return False, 0, 0, 0, 0, 0

    robot = Robot(path, speed=2)
    replan_count = 0
    total_replan_time = 0
    near_collision_count = 0

    ideal_distance = np.linalg.norm(np.array(goal) - np.array(start))
    traveled_distance = 0

    simulation_ticks = 0
    max_ticks = 5000  # timeout to avoid infinite loops

    running = True
    while running and simulation_ticks < max_ticks:
        simulation_ticks += 1
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return False, 0, 0, 0, 0, 0

        for mob in moving_obstacles:
            mob.update()

        obstruction_index = planner.is_path_obstructed(robot.current_index)
        if obstruction_index is not None:
            replanning_start = time.time()
            planner = RRTStar(tuple(robot.position), goal, obstacles, temporary_obstacles=[mob.polygon for mob in moving_obstacles])
            new_path = planner.find_path()
            replanning_end = time.time()

            if new_path is None:
                return False, replan_count, total_replan_time, traveled_distance, ideal_distance, near_collision_count

            robot.path = [(robot.position[0], robot.position[1])] + new_path[1:]
            robot.current_index = 0
            replan_count += 1
            total_replan_time += replanning_end - replanning_start

        robot.update()

        traveled_distance += robot.speed

        # Check near collisions
        robot_shape = Point(robot.position).buffer(ROBOT_RADIUS)
        for mob in moving_obstacles:
            if robot_shape.distance(mob.polygon) < ROBOT_RADIUS * 1.5:
                near_collision_count += 1
                break

        if robot.reached_goal():
            return True, replan_count, total_replan_time, traveled_distance, ideal_distance, near_collision_count

    return False, replan_count, total_replan_time, traveled_distance, ideal_distance, near_collision_count

# Main runner
tracker = MetricsTracker()

for run_id in range(50):
    print(f"Running Simulation {run_id+1}/50...")
    success, replans, replan_time, path_len, ideal_dist, near_col = run_simulation()
    tracker.record_run(success, replans, replan_time, path_len, ideal_dist, near_col)

tracker.report(50)

pygame.quit()
