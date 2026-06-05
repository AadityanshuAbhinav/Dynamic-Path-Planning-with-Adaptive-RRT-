import numpy as np
import random
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

class A_star:

    def __init__(self):
        self.start = None
        self.goal = None
        self.width = None
        self.height = None
        self.static_obs_list = []

    def equation(self, a, b):
        if a[0] == b[0]:
            return [0, 1, -a[0]]
        else:
            m = (b[1] - a[1]) / (b[0] - a[0])
            c = b[1] - m * b[0]
            return [m, -1, c]

    def check(self, current, destination, obs_list):
        for poly in obs_list:
            for j in range(len(poly)):
                b = poly[j]
                b_next = poly[(j + 1) % len(poly)]
                m, m_y, c = self.equation(current, destination)
                d_1 = m * b[0] + m_y * b[1] + c
                d_2 = m * b_next[0] + m_y * b_next[1] + c
                if (d_1 <= 0 and d_2 >= 0) or (d_2 <= 0 and d_1 >= 0):
                    return 0
        return 1

    def dist(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def check_point(self, coord, obs_list):
        p = Point(coord[0], coord[1])
        for poly in obs_list:
            polygon = Polygon(poly)
            if p.within(polygon):
                return True
        return False

    def generate_points(self, N):
        points = set()
        for _ in range(20 * N):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            coord = (x, y)
            if not self.check_point(coord, self.static_obs_list):
                points.add(coord)
            if len(points) >= N:
                break
        return list(points)

    def knn(self, point, points, k):
        neighbors = []
        for p in points:
            if p != point:
                if self.check(point, p, self.static_obs_list):
                    if self.dist(point, p) <= k:
                        neighbors.append(p)
        return neighbors

    def create_graph(self, obs_list, points, dist_threshold, start, goal):
        graph = {}
        all_points = points + [start, goal]
        for i in all_points:
            graph[i] = self.knn(i, all_points, dist_threshold)
        return graph

    def euclidean(self, point, goal):
        return self.dist(point, goal)

    def plan_path(self, graph, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        path = []
        fringe = [[0, [start, [], 0]]]
        closed = []

        while len(fringe) != 0:
            fringe.sort()
            a = fringe.pop(0)
            cost_sum, node, history, g = a[0], a[1][0], a[1][1], a[1][2]
            closed.append(node)

            if node == goal:
                path = history + [node]
                break

            neighbors = graph[node]

            for i in neighbors:
                if i not in closed:
                    g_new = g + 1
                    h_new = self.euclidean(i, goal)
                    cost_sum_new = g_new + h_new
                    count = 0

                    for j in fringe:
                        if j[1][0] == i and j[1][2] > g_new:
                            j[1][2] = g_new
                            j[0] = g_new + h_new
                            j[1][1] = history + [node]
                            count += 1
                        if j[1][0] == i and j[1][2] <= g_new:
                            count += 1

                    if count == 0:
                        fringe.append([cost_sum_new, [i, history + [node], g_new]])

        return path

    def adaptive_a_star(self, graph, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        current_start = start
        final_path = []

        while True:
            path = self.plan_path(graph, current_start, goal)

            if not path:
                print("No path found!")
                return final_path

            collision_found = False
            for idx in range(len(path) - 1):
                p1 = path[idx]
                p2 = path[idx + 1]

                if not self.check(p1, p2, self.static_obs_list):
                    # Obstacle encountered between p1 and p2
                    final_path += path[:idx + 1]
                    current_start = p1
                    collision_found = True
                    break

            if not collision_found:
                # No collision found; add full path
                final_path += path
                break

        return final_path


if __name__ == "__main__":
    A = A_star()
    
    A.start = (10, 10)
    A.goal = (490, 490)
    A.width = 500
    A.height = 500
    A.static_obs_list = [
        [(100,100),(150,100),(150,150),(100,150)],
        [(300,300),(400,300),(400,400),(300,400)],
        [(200,0),(250,0),(250,250),(200,250)]
    ]

    points = A.generate_points(300)
    graph = A.create_graph(A.static_obs_list, points, dist_threshold=100, start=A.start, goal=A.goal)

    adaptive_path = A.adaptive_a_star(graph, A.start, A.goal)
    
    print("Planned adaptive path:")
    print(adaptive_path)

    # ------- PLOTTING -------
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot obstacles
    for obs in A.static_obs_list:
        obs_poly = Polygon(obs)
        x, y = obs_poly.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.7)

    # Plot path
    if adaptive_path:
        path_x = [p[0] for p in adaptive_path]
        path_y = [p[1] for p in adaptive_path]
        ax.plot(path_x, path_y, '-or', label='Path')

    # Plot start and goal
    ax.plot(A.start[0], A.start[1], 'go', markersize=10, label='Start')
    ax.plot(A.goal[0], A.goal[1], 'rx', markersize=10, label='Goal')

    ax.set_xlim(0, A.width)
    ax.set_ylim(0, A.height)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.title("Adaptive A* Path Planning")
    plt.show()
