#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
import numpy as np
import pygame
import math
import random
from shapely.geometry import Polygon, LineString
from shapely.geometry import Point as Pointz
from tf_transformations import euler_from_quaternion

# Constants
WIDTH, HEIGHT = 700, 700
SCALE = 70
ROBOT_RADIUS = 10
STEP_SIZE = 15
MAX_ITER = 1500
GOAL_SAMPLE_RATE = 0.05
NEIGHBOR_RADIUS = 40

class NodeRRT:
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
        self.position = np.array([self.center[0] - path_size/2, self.center[1] - path_size/2], dtype=np.float64)

    def update(self):
        current_time = pygame.time.get_ticks()
        if "pause" in self.state:
            if current_time - self.last_switch_time >= self.pause_duration:
                if self.state == "pause_right": self.state = "moving_down"
                elif self.state == "pause_down": self.state = "moving_left"
                elif self.state == "pause_left": self.state = "moving_up"
                elif self.state == "pause_up": self.state = "moving_right"
            else:
                return
        if self.state == "moving_right":
            self.position[0] += self.speed
            if self.position[0] >= self.center[0] + self.path_size/2:
                self.position[0] = self.center[0] + self.path_size/2
                self.state = "pause_right"
                self.last_switch_time = current_time
        elif self.state == "moving_down":
            self.position[1] += self.speed
            if self.position[1] >= self.center[1] + self.path_size/2:
                self.position[1] = self.center[1] + self.path_size/2
                self.state = "pause_down"
                self.last_switch_time = current_time
        elif self.state == "moving_left":
            self.position[0] -= self.speed
            if self.position[0] <= self.center[0] - self.path_size/2:
                self.position[0] = self.center[0] - self.path_size/2
                self.state = "pause_left"
                self.last_switch_time = current_time
        elif self.state == "moving_up":
            self.position[1] -= self.speed
            if self.position[1] <= self.center[1] - self.path_size/2:
                self.position[1] = self.center[1] - self.path_size/2
                self.state = "pause_up"
                self.last_switch_time = current_time

    @property
    def polygon(self):
        return Pointz(self.position).buffer(self.radius)

class RRTStar:
    def __init__(self, start, goal, obstacles):
        self.start = NodeRRT(*start)
        self.goal = NodeRRT(*goal)
        self.nodes = [self.start]
        self.obstacles = obstacles
        self.path = None

    def get_random_node(self):
        if random.random() < GOAL_SAMPLE_RATE:
            return self.goal
        return NodeRRT(random.randint(0, WIDTH), random.randint(0, HEIGHT))

    def get_nearest_node(self, node):
        return min(self.nodes, key=lambda n: (n.x - node.x)**2 + (n.y - node.y)**2)

    def steer(self, from_node, to_node):
        dx, dy = to_node.x - from_node.x, to_node.y - from_node.y
        dist = np.hypot(dx, dy)
        if dist == 0: return from_node
        scale = min(STEP_SIZE / dist, 1.0)
        return NodeRRT(from_node.x + dx * scale, from_node.y + dy * scale)

    def check_collision(self, start, end):
        line = LineString([start, end]).buffer(ROBOT_RADIUS)
        for obs in self.obstacles:
            if line.intersects(obs):
                return True
        return False

    def is_point_valid(self, point):
        p = Pointz(point)
        circle = p.buffer(ROBOT_RADIUS)
        for obs in self.obstacles:
            if circle.intersects(obs):
                return False
        return True

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

            if np.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) < STEP_SIZE:
                if not self.check_collision((new_node.x, new_node.y), (self.goal.x, self.goal.y)):
                    self.goal.parent = new_node
                    self.path = self.extract_path()
                    return self.path

        return None

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

class PositionDisplay(Node):
    def __init__(self):
        super().__init__('dynamic_rrt_star_planner')

        self.robot_sub = self.create_subscription(Odometry, '/odom', self.robot_callback, 10)
        self.actor_sub = self.create_subscription(Point, '/actor_position', self.actor_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.robot_pos = [0.0, 0.0, 0.0]  # x, y, yaw
        self.actor_pos = [0.0, 0.0]
        self.robot_ready = False
        self.goal = (WIDTH-50, HEIGHT-50)

        self.moving_obstacles = [CircularMover(center=(WIDTH//2, HEIGHT//2), path_size=300, radius=20, speed=2, pause_duration=1)]

        self.path = []
        self.path_index = 0

        pygame.init()
        self.clock = pygame.time.Clock()

    def robot_callback(self, msg):
        self.robot_pos[0] = 700 - msg.pose.pose.position.x * SCALE
        self.robot_pos[1] = msg.pose.pose.position.y * SCALE
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.robot_pos[2] = euler_from_quaternion(orientation_list)
        self.robot_ready = True

    def actor_callback(self, msg):
        self.actor_pos = [
            700 - msg.x * SCALE,
            msg.y * SCALE
        ]

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            for mob in self.moving_obstacles:
                mob.update()

            if self.robot_ready:
                if not self.path or self.path_index >= len(self.path):
                    obstacles = [
                        Polygon([
                            (self.actor_pos[0]-20, self.actor_pos[1]-20),
                            (self.actor_pos[0]+20, self.actor_pos[1]-20),
                            (self.actor_pos[0]+20, self.actor_pos[1]+20),
                            (self.actor_pos[0]-20, self.actor_pos[1]+20)
                        ])
                    ]
                    for mob in self.moving_obstacles:
                        obstacles.append(mob.polygon)

                    rrt = RRTStar(self.robot_pos[:2], self.goal, obstacles)
                    new_path = rrt.find_path()
                    if new_path:
                        self.path = new_path
                        self.path_index = 0

                if self.path and self.path_index < len(self.path):
                    target = self.path[self.path_index]
                    if self.move_toward(target):
                        self.path_index += 1

            self.clock.tick(30)

    def move_toward(self, target_pos):
        cmd = Twist()
        rx, ry, ryaw = self.robot_pos
        tx, ty = target_pos

        dx, dy = tx - rx, ty - ry
        dist = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - ryaw

        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        if abs(angle_diff) > 0.1:
            cmd.angular.z = 1.5 * angle_diff
        else:
            cmd.linear.x = min(0.5, dist/100)
        
        self.cmd_vel_pub.publish(cmd)
        return dist < 10

def main(args=None):
    rclpy.init(args=args)
    node = PositionDisplay()
    node.run()

if __name__ == "__main__":
    main()
