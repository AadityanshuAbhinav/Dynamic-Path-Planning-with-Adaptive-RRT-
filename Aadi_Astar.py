import pygame
import random
import math
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import KDTree
import time

# Define constants
WIDTH, HEIGHT = 500, 500
WHITE = (255, 255, 255)
GREY = (160, 160, 160)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Path Planning with Dynamic Obstacle")
clock = pygame.time.Clock()

def is_collision_free_line(p1, p2, obs_list):
    line = LineString([p1, p2])
    for obs in obs_list:
        poly = Polygon(obs)
        if line.crosses(poly) or line.within(poly) or line.intersects(poly):
            return False
    return True

class A_star:
    def __init__(self):
        self.start = (10, 10)
        self.goal = (490, 490)
        self.width = WIDTH
        self.height = HEIGHT
        self.static_obs_list = [
            [(100, 100), (150, 100), (150, 150), (100, 150)],  
            [(300, 300), (400, 300), (400, 400), (300, 400)],  
            [(200, 0), (250, 0), (250, 250), (200, 250)]       
        ]

    def dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def check_point(self, coord, obs_list):
        p = Point(coord[0], coord[1])
        for obs in obs_list:
            poly = Polygon(obs)
            if p.within(poly):
                return True
        return False

    def generate_points(self, N):
        points = []
        for _ in range(20 * N):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            if not self.check_point((x, y), self.static_obs_list):
                points.append((x, y))
            if len(points) >= N:
                break
        return points

    def create_graph(self, obs_list, points, dist_threshold, start, goal):
        graph = {}
        all_points = points + [start, goal]
        
        for i in all_points:
            neighbors = []
            for j in all_points:
                if i != j and self.dist(i, j) <= dist_threshold:
                    if is_collision_free_line(i, j, obs_list):
                        neighbors.append(j)
            graph[i] = neighbors

        return graph

    def euclidean(self, point, goal):
        return self.dist(point, goal)

    def plan_path(self, graph, start, goal):
        path = []
        fringe = [[0, [start, [], 0]]]
        closed = set()

        while fringe:
            fringe.sort(key=lambda x: x[0]) 
            cost_sum, current_state = fringe.pop(0)
            current_node, history, g = current_state

            if current_node == goal:
                path = history + [current_node]
                break

            closed.add(current_node)
            neighbors = graph.get(current_node, [])

            for neighbor in neighbors:
                if neighbor not in closed:
                    g_new = g + self.dist(current_node, neighbor)
                    h_new = self.euclidean(neighbor, goal)
                    cost_sum_new = g_new + h_new

                    found_in_fringe = False
                    for i, item in enumerate(fringe):
                        fringe_node = item[1][0]
                        if fringe_node == neighbor and item[1][2] > g_new:
                            fringe[i] = [cost_sum_new, [neighbor, history + [current_node], g_new]]
                            found_in_fringe = True
                            break

                    if not found_in_fringe:
                        fringe.append([cost_sum_new, [neighbor, history + [current_node], g_new]])

        return path

def draw_static_obs(static_obs_list):
    for obs in static_obs_list:
        pygame.draw.polygon(screen, GREY, obs)

def draw_robot(pos, color=GREEN, radius=5):
    pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), radius)

def draw_path(path, color=RED):
    if path:
        for i in range(len(path)-1):
            pygame.draw.line(screen, color, path[i], path[i+1], 2)

def move_obstacle(pos, step_size=2):
    x, y, dir_idx = pos
    dirs = [(1,0), (0,1), (-1,0), (0,-1)]  
    dx, dy = dirs[dir_idx]
    x += dx * step_size
    y += dy * step_size

    if x >= 450 and dir_idx == 0:
        dir_idx = 1
    if y >= 450 and dir_idx == 1:
        dir_idx = 2
    if x <= 300 and dir_idx == 2:
        dir_idx = 3
    if y <= 300 and dir_idx == 3:
        dir_idx = 0

    return (x, y, dir_idx)

def check_collision_path(path, obs_center, obs_radius):
    for p in path:
        if math.hypot(p[0]-obs_center[0], p[1]-obs_center[1]) <= obs_radius + 5:
            return True
    return False

if __name__ == "__main__":
    A = A_star()
    points = A.generate_points(300)
    graph = A.create_graph(A.static_obs_list, points, dist_threshold=100, start=A.start, goal=A.goal)
    path = A.plan_path(graph, A.start, A.goal)
    if not path:
        print("No initial path found!")
        exit()

    # Dynamic obstacle init
    dyn_obs = (300, 300, 0)
    dyn_radius = 15

    running = True
    idx = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        draw_static_obs(A.static_obs_list)
        pygame.draw.circle(screen, BLUE, (int(dyn_obs[0]), int(dyn_obs[1])), dyn_radius)

        draw_path(path)

        if idx < len(path):
            robot_pos = path[idx]
            draw_robot(robot_pos)

            future_path = path[idx:]
            if check_collision_path(future_path, (dyn_obs[0], dyn_obs[1]), dyn_radius):
                print("Obstacle detected on path! Replanning...")
                temp_obs = A.static_obs_list + [[
                    (dyn_obs[0]-dyn_radius, dyn_obs[1]-dyn_radius),
                    (dyn_obs[0]+dyn_radius, dyn_obs[1]-dyn_radius),
                    (dyn_obs[0]+dyn_radius, dyn_obs[1]+dyn_radius),
                    (dyn_obs[0]-dyn_radius, dyn_obs[1]+dyn_radius)
                ]]
                graph = A.create_graph(temp_obs, points, dist_threshold=100, start=robot_pos, goal=A.goal)
                new_path = A.plan_path(graph, robot_pos, A.goal)
                if new_path:
                    path = path[:idx] + new_path
                else:
                    print("No new path found!")
            
            idx += 1
        else:
            draw_robot(A.goal, color=YELLOW)
        
        dyn_obs = move_obstacle(dyn_obs)

        pygame.display.update()
        clock.tick(30)

    pygame.quit()
