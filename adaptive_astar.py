import random
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import numpy as np
from scipy.spatial import KDTree

class A_star:
    def __init__(self):
        self.start = (10, 10)
        self.goal = (490, 490)
        self.width = 500
        self.height = 500
        self.static_obs_list = [
            [(100, 100), (150, 100), (150, 150), (100, 150)],  # Static obstacle 1
            [(300, 300), (400, 300), (400, 400), (300, 400)],  # Static obstacle 2
            [(200, 0), (250, 0), (250, 250), (200, 250)]       # Static obstacle 3
        ]

    def equation(self, a, b):
        if a[0] == b[0]:
            return [0, 1, -a[0]]
        else:
            m = (b[1] - a[1]) / (b[0] - a[0])
            c = b[1] - m * b[0]
            return [m, -1, c]

    def check(self, current, destination, obs_list):
        for i in obs_list:
            for j in range(len(i)):
                b = i[j]
                if j == len(i) - 1:
                    b_next = i[0]
                else:
                    b_next = i[j + 1]
                m, m_y, c = self.equation(current, destination)
                d_1 = m * b[0] + m_y * b[1] + c
                d_2 = m * b_next[0] + m_y * b_next[1] + c
                if (d_1 <= 0 and d_2 >= 0) or (d_2 <= 0 and d_1 >= 0):
                    return 0
                else:
                    return 1

    def dist(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def check_point(self, coord, obs_list):
        flag = False
        for i in obs_list:
            p = Point(coord[0], coord[1])
            poly = Polygon(i)
            new = p.within(poly)
            if new != flag:
                flag = new
                break
        return flag

    def generate_points(self, N):
        points = []
        for i in range(20 * N):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            flag = self.check_point((x, y), self.static_obs_list)
            if flag == False and (x, y) not in points:
                points.append((x, y))  # store as tuple
        return points

    def knn(self, point, points, dist_threshold):
        kdtree = KDTree(points)
        neighbors = kdtree.query_ball_point(point, dist_threshold)
        return neighbors


    def create_graph(self, obs_list, points, dist_threshold, start, goal):
        graph = {}
        for i in points:
            neighbor_indices = self.knn(i, points, dist_threshold)
            neighbors = [points[idx] for idx in neighbor_indices]
            graph[i] = self.knn(i, points, dist_threshold)
        neighbor_indices_start = self.knn(start, points, dist_threshold)
        graph[start] = [points[idx] for idx in neighbor_indices_start]
        neighbor_indices_goal = self.knn(goal, points, dist_threshold)
        graph[goal] = [points[idx] for idx in neighbor_indices_goal]
        return graph

    def euclidean(self, point, goal):
        print(f"Current Node: {current_node}, Type: {type(current_node)}")
        print(f"Neighbor: {neighbor}, Type: {type(neighbor)}")

        return self.dist(point, goal)

    def plan_path(self, graph, start, goal):
        path = []
        fringe = [[0, [start, [], 0]]]
        closed = set()  # Use a set for faster checking

        while fringe:
            fringe.sort(key=lambda x: x[0])  # Sort by cost
            cost_sum, current_state = fringe.pop(0)
            current_node, history, g = current_state[0], current_state[1], current_state[2]

            if current_node == goal:
                path = history + [current_node]
                break

            closed.add(current_node)

            neighbors = graph.get(current_node, [])  # Handle cases where a node might not have neighbors

            for neighbor in neighbors:
                if neighbor not in closed:
                    g_new = g + self.dist(current_node, neighbor)
                    h_new = self.euclidean(goal, neighbor)
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

    def adaptive_a_star(self, graph, start, goal):
        current_start = start
        path = self.plan_path(graph, current_start, goal)
        if not path:
            return []

        # Test adaptiveness by introducing a new obstacle
        dynamic_obstacle = [(300, 350), (350, 350), (350, 400), (300, 400)]  # Moving obstacle
        self.static_obs_list.append(dynamic_obstacle)  # Add obstacle

        # Replan after obstacle insertion
        new_graph = self.create_graph(self.static_obs_list, list(graph.keys()), dist_threshold=100, start=start, goal=goal)
        print(type(graph.keys()))  # Check if it's dict_keys
        print(graph.keys())  # See the actual keys stored
        new_path = self.plan_path(new_graph, current_start, goal)
        
        if new_path:
            print("Replanning successful.")
            return new_path
        else:
            print("No valid path found after replanning.")
            return path


if __name__ == "__main__":
    A = A_star()
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
