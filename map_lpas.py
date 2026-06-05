import pygame
import numpy as np
import heapq

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 700, 700
GRID_SIZE = 20
ROWS, COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE

display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('LPA* Path Planning')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (200, 200, 200)

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class LPAStar:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.g = {start: float('inf')}
        self.rhs = {start: 0}
        self.queue = []
        self.obstacles = set()
        self.initialize()

    def initialize(self):
        for r in range(ROWS):
            for c in range(COLS):
                self.g[(r, c)] = float('inf')  # Default g-value
                self.rhs[(r, c)] = float('inf')  # Default rhs-value
        self.rhs[self.start] = 0  # Start vertex has rhs = 0
        heapq.heappush(self.queue, (self.calculate_key(self.start), self.start))



    def calculate_key(self, vertex):
        g_rhs_min = min(self.g[vertex], self.rhs[vertex])
        return (g_rhs_min + heuristic(vertex, self.goal), g_rhs_min)

    def update_vertex(self, vertex):
        if vertex != self.start:
            neighbors = self.get_neighbors(vertex)
            new_rhs = min([self.g[n] + 1 for n in neighbors if n not in self.obstacles])
            if new_rhs != self.rhs[vertex]:
                self.rhs[vertex] = new_rhs
                if vertex in self.queue:
                    self.queue.remove(vertex)
                    heapq.heapify(self.queue)
                if self.g[vertex] != self.rhs[vertex]:
                    heapq.heappush(self.queue, (self.calculate_key(vertex), vertex))




    def compute_shortest_path(self, max_steps=1000):
        steps = 0
        while steps < max_steps and self.queue and (self.calculate_key(self.goal) > self.queue[0][0]):
            current_vertex = heapq.heappop(self.queue)[1]
            print(f"Processing vertex: {current_vertex}")
            print(f"g[{current_vertex}] = {self.g[current_vertex]}, rhs[{current_vertex}] = {self.rhs[current_vertex]}")

            if current_vertex not in self.g or current_vertex not in self.rhs:
                continue  # Skip invalid vertices

            if self.g[current_vertex] > self.rhs[current_vertex]:
                # Overconsistent case
                self.g[current_vertex] = self.rhs[current_vertex]
                print(f"Updated g[{current_vertex}] to {self.g[current_vertex]} (Overconsistent)")
                for neighbor in self.get_neighbors(current_vertex):
                    self.update_vertex(neighbor)
            else:
                # Underconsistent case
                self.g[current_vertex] = float('inf')
                print(f"Set g[{current_vertex}] to infinity (Underconsistent)")
                for neighbor in [current_vertex] + self.get_neighbors(current_vertex):
                    self.update_vertex(neighbor)
            steps += 1
        if steps == max_steps:
            print("Max steps reached. Terminating search.")




    def get_neighbors(self, vertex):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for d in directions:
            neighbor = (vertex[0] + d[0], vertex[1] + d[1])
            if 0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS:
                neighbors.append(neighbor)
        return neighbors

    def update_obstacle(self, obstacle_pos):
        if obstacle_pos not in self.obstacles:
            self.obstacles.add(obstacle_pos)
        else:
            self.obstacles.remove(obstacle_pos)
        for neighbor in self.get_neighbors(obstacle_pos):
            self.update_vertex(neighbor)


def draw_grid(surface):
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * GRID_SIZE, r * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(surface, GREY, rect, 1)

def draw_obstacles(surface, obstacles):
    for obs in obstacles:
        rect = pygame.Rect(obs[1] * GRID_SIZE, obs[0] * GRID_SIZE,
                           GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, BLACK, rect)

def draw_path(surface, path):
    for p in path:
        rect = pygame.Rect(p[1] * GRID_SIZE + GRID_SIZE // 4,
                           p[0] * GRID_SIZE + GRID_SIZE // 4,
                           GRID_SIZE // 2,
                           GRID_SIZE // 2)
        pygame.draw.rect(surface, RED, rect)

def main():
    # Define start and goal positions
    start = (ROWS - 2, 2)  # Bottom-left corner
    goal = (2, COLS - 2)  # Top-right corner
    print("main")

    # Initialize LPA* algorithm
    lpa_star = LPAStar(start, goal)

    running = True
    path = []
    clock = pygame.time.Clock()

    # Add some initial obstacles
    for i in range(10, 30):
        lpa_star.update_obstacle((i, i))

    # lpa_star.obstacles = set()
    while running:
        print("Now running")
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False2
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                grid_col = x // GRID_SIZE  # Column (x-axis)
                grid_row = y // GRID_SIZE  # Row (y-axis)
                lpa_star.update_obstacle((grid_row, grid_col))


        # Process a few steps of LPA* per frame
        lpa_star.compute_shortest_path(max_steps=50)  # Adjust steps for performance

        # Extract the path from start to goal
        path = []
        current = goal
        while current != start:
            path.append(current)
            neighbors = lpa_star.get_neighbors(current)
            valid_neighbors = [n for n in neighbors if n in lpa_star.g and lpa_star.g[n] != float('inf') and n not in lpa_star.obstacles]
            print(f"Current: {current}, Valid neighbors: {valid_neighbors}")

            if not valid_neighbors:
                print(f"No valid neighbors for {current}. Path blocked!")
                path = []
                break

            current = min(valid_neighbors, key=lambda n: lpa_star.g[n])
            print(f"Moving to next node: {current}")

        if current == start:
            path.append(start)
            path.reverse()
        else:
            path = []


        path.append(start)
        path.reverse()

        # Visualization
        display.fill(WHITE)
        draw_grid(display)
        draw_obstacles(display, lpa_star.obstacles)
        draw_path(display, path)

        pygame.draw.circle(display, BLUE, (start[1] * GRID_SIZE + GRID_SIZE // 2,
                                           start[0] * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 3)
        pygame.draw.circle(display, GREEN, (goal[1] * GRID_SIZE + GRID_SIZE // 2,
                                            goal[0] * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 3)

        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
