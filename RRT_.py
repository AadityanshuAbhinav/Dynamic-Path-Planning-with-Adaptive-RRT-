import numpy as np
import pygame
from shapely.geometry import Point, Polygon
import random
import time

pygame.init()
WIDTH, HEIGHT = 700, 700
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pioneer 3-AT RRT* Path Planning')

class RRTStar:

    def __init__(self):
        self.start = (1, 1)
        self.goal = (1, 1)
        self.tree = {self.start: [self.start, 0]}
        self.obs_list = []
        self.time_start = time.time()
        self.velocity = 1
    
    def t_obs(self, t):
        obs = []
        return obs
    
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
    
    def sample(self):
        return (np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT))
    
    def dist(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def knn(self, tree, pt, obs_list):
        knn_pts= []
        keys = list(tree.keys())
        for i in keys:
            if self.check(i, pt, obs_list) == 1:
                knn_pts.append([self.dist(i, pt), i])
        if len(keys) >= 6:
            knn_pts.sort()
            knn_pts = knn_pts[:6]
        return knn_pts
    
    def rewire(self, knn_pts, tree, pt):
        cost = []
        for i in knn_pts:
            cost.append(i[0] + tree[i[1]][1])
        ind = cost.index(min(cost))
        tree[pt] = [knn_pts[ind], cost[ind]]

        for i in knn_pts:
            if i != knn_pts[ind]:
                if tree[pt][1] + knn_pts[i][0] <= tree[i][1]:
                    del tree[i]
                    tree[i] = [pt, tree[pt][1] + knn_pts[i][0]] 
        return tree
    
    def goal_check(self, tree, obs_list):
        keys = list(tree.keys())
        min = WIDTH
        for i in keys:
            if self.check(i, self.goal, obs_list) == 1:
                if self.dist(i, self.goal) <= 100:
                    tree[self.goal] = [i, tree[i][1] + self.dist(i, self.goal)]
                    return tree, True
        return tree, False

    def add_to_tree(self, obs_list, tree):
        new_point = self.sample()
        if self.check_point(new_point, obs_list) == False and new_point not in tree and new_point != self.start:
            knn_pts = self.knn(tree, new_point)
            tree = self.rewire(knn_pts, tree, new_point)

    def extract_path(self, tree):
        path = [self.goal]
        cost = []
        while True:
            a = path[-1]
            path.append(tree[a][0])
            cost.append(tree[a][1])

            if tree[a][0] == self.start:
                break
        return path
    
    def path_dist(self, path):
        dist = 0
        for i in range(len(path) - 1):
            dist += np.sqrt((path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2)
        return dist

    
    def check_time(self, new_time, path): 
        seg = len(path) #how many segments you want to divide this time into
        time_taken = self.path_dist(path) / self.velocity #enter time taken for robot acc to path planned by rrt*
        t = np.linspace(0, time_taken, seg)
        for i in t:
            cur_pt = (1, 1) #current position of obstacle
            obs = self.t_obs(new_time)
            flag = self.check_point(cur_pt, obs)
            if flag == True:
                return cur_pt
                break
            else:
                return True
    def main(self):
        while True:
            new_time = time.time() - self.time_start
            if self.goal_check(self.tree, self.obs_list):
                path = self.extract_path(self.tree)
                break
            self.add_to_tree(self.obs_list, self.tree)
            if self.check_time(new_time, path) != True:
                cur_pt = self.check_time(new_time, path)
                new_start = (cur_pt[0], cur_pt[1])
                self.start = new_start
                self.tree = {self.start: [self.start, 0]}
                path = path[:]
            else:
                break
            
