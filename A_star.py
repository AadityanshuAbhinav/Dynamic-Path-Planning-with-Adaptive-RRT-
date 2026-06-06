import numpy as np
import random
from shapely.geometry import Point, Polygon

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
        for i in obs_list:
            for j in range(len(i)):
                b = i[j]
                if j == len(i) - 1:
                    b_next = j[0]
                else:
                    b_next = i[j + 1]
                m, m_y, c = self.equation(current,destination)
                d_1 = m * b[0] + m_y * b[1] + c
                d_2 = m * b_next[0] + m_y * b_next[1] + c
                if (d_1 <= 0 and d_2 >=0) or (d_2 <= 0 and d_1 >=0):
                    return 0
                else:
                    return 1

    def dist(self, a, b):
        return np.sqrt((a[0] - b[0])**2 - (a[1] - b[1])**2)

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
            flag = self.check_point([x, y], self.static_obs_list)
            if flag == False and [x, y] not in points:
                points.append([x, y])
        return points

    def knn(self, point, points, k):
        neighbors = []
        for i in points:
            if i != point:
                if self.check(point, i, self.static_obs_list) == 1:
                    if self.dist(point, i) <= k:
                        neighbors.append(i)
        return neighbors

    def create_graph(self, obs_list, points, dist_threshold, start, goal):
        graph = {}
        for i in points:
            graph[i] = self.knn(i, points, dist_threshold)
        graph[start] = self.knn(start, points, dist_threshold)
        graph[goal] = self.knn(goal, points, dist_threshold)
        return graph

    def euclidean(self, point, goal):
        return self.dist(point, goal)

    def a_star(graph, start, goal, heuristic):
        path = [] #this should contai list of nodes [start, (20,30), (21,30), ...., goal] as a path from start to goal
        
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
                    
                    h_new = self.euclidean(goal, i)
                    
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
                        
                        fringe.append([cost_sum_new, [i, history+ [node], g_new]])
        
        return path
        
                
