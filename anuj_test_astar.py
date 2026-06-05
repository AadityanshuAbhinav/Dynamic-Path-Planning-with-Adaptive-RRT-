import numpy as np
import heapq
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import Point, Polygon

class AdaptiveAStar:
    def __init__(self):
        self.start = (10, 10)
        self.goal = (490, 490)
        self.width = 500
        self.height = 500
        self.static_obs_list = [
            [(100, 100), (150, 100), (150, 150), (100, 150)],  
            [(300, 300), (400, 300), (400, 400), (300, 400)],  
            [(200, 0), (250, 0), (250, 250), (200, 250)]      
        ]
        self.moving_obstacle = [(300, 350), (350, 350), (350, 400), (300, 400)]  
        self.path = []

    def dist(self, a, b):
        """ Computes Euclidean distance with validation """
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def heuristic(self, point, goal):
        """ Manhattan distance heuristic for better convergence """
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def check_point(self, coord, obs_list):
        """ Check if a point is inside any obstacle """
        for obs in obs_list:
            if Point(coord).within(Polygon(obs)):
                return True
        return False

    def generate_points(self, N):
        """ Generate random points while avoiding obstacles """
        points = []
        for _ in range(10 * N):  
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            if not self.check_point((x, y), self.static_obs_list + [self.moving_obstacle]):
                points.append((x, y))
        return points

    def equation(self, a, b):
        if a[0] == b[0]:
            return [0, 1, -a[0]]
        else:
            m = (b[1] - a[1]) / (b[0] - a[0])
            c = b[1] - m * b[0]
            return [m, -1, c]

    def check(self, current, destination, obs_list):
        flag = 1
        for i in obs_list:
            for j in range(len(i)):
                b = i[j]
                if j == len(i) - 1:
                    b_next = i[0]
                else:
                    b_next = i[j + 1]
                m, m_y, c = self.equation(current,destination)
                d_1 = m * b[0] + m_y * b[1] + c
                d_2 = m * b_next[0] + m_y * b_next[1] + c
                if (d_1 <= 0 and d_2 >=0) or (d_2 <= 0 and d_1 >=0):
                    flag = 0
                    break
        return flag

    def create_graph(self, points, dist_threshold):
        """ Connect nodes using a K-nearest neighbors approach """
        graph = {p: [] for p in points}
        for p in points:
            neighbors = [q for q in points if self.dist(p, q) <= dist_threshold and p != q and self.check(p, q, self.static_obs_list) == 1]
            graph[p] = neighbors
        return graph

    def a_star(self, graph, start, goal):
        """ A* path planning with priority queue optimization """
        if start not in graph or goal not in graph:
            return []

        fringe = []
        heapq.heappush(fringe, (self.heuristic(start, goal), start, [], 0))  
        closed = set()

        while fringe:
            cost_sum, node, history, g = heapq.heappop(fringe)  
            closed.add(node)

            if node == goal:
                return history + [node]

            for neighbor in graph.get(node, []):
                if neighbor not in closed and neighbor not in [item[1] for item in fringe]:
                    g_new = g + self.dist(node, neighbor)
                    h_new = self.heuristic(neighbor, goal)
                    heapq.heappush(fringe, (g_new + h_new, neighbor, history + [node], g_new))

        return []

    def detect_collision(self):
        """ Identify the last collision-free point before obstruction """
        for i, point in enumerate(self.path):
            if self.check_point(point, self.static_obs_list + [self.moving_obstacle]):
                return self.path[:i]  
        return self.path

    def adaptive_a_star(self):
        """ Run adaptive path planning with dynamic obstacle handling and visualization """
        points = self.generate_points(200)
        graph = self.create_graph(points + [self.start, self.goal], dist_threshold=50)
        self.path = self.a_star(graph, self.start, self.goal)

        if not self.path:
            print("Initial path planning failed.")
            return []

        # Animation setup
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame):
            ax.clear()

            # Move obstacle incrementally
            shift = frame * 2
            self.moving_obstacle = [(320 + shift, 350), (370 + shift, 350), (370 + shift, 400), (320 + shift, 400)]

            # Detect collisions and update path
            valid_path_segment = self.detect_collision()
            last_safe_point = valid_path_segment[-1] if valid_path_segment else self.start

            graph = self.create_graph(points + [last_safe_point, self.goal], dist_threshold=50)
            new_path = self.a_star(graph, last_safe_point, self.goal)

            if new_path:
                self.path = valid_path_segment + new_path  

            # Plot static obstacles
            for obs in self.static_obs_list:
                obs_poly = Polygon(obs)
                x, y = obs_poly.exterior.xy
                ax.fill(x, y, color='gray', alpha=0.7)

            # Plot moving obstacle
            obs_poly = Polygon(self.moving_obstacle)
            x, y = obs_poly.exterior.xy
            ax.fill(x, y, color='red', alpha=0.5, label="Moving Obstacle")

            # Plot path
            if self.path:
                path_x = [p[0] for p in self.path]
                path_y = [p[1] for p in self.path]
                ax.plot(path_x, path_y, '-or', label="Path")

            ax.plot(self.start[0], self.start[1], 'go', markersize=10, label="Start")
            ax.plot(self.goal[0], self.goal[1], 'rx', markersize=10, label="Goal")

            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()
            plt.title(f"Frame {frame}: Adaptive A*")

        ani = animation.FuncAnimation(fig, update, frames=20, interval=500)
        plt.show()
        return self.path

if __name__ == "__main__":
    A = AdaptiveAStar()
    final_path = A.adaptive_a_star()

    if not final_path:
        print("Final Path: No valid path found.")
    else:
        print("Final Path:", final_path)
