# import pygame module in this program 
import pygame 

# activate the pygame library . 
# initiate pygame and give permission 
# to use pygame's functionality. 
pygame.init() 

# define the RGB value 
# for white, green, 
# blue, black, red 
# colour respectively. 
white = (255, 255, 255) 
green = (0, 255, 0) 
blue = (0, 0, 128) 
black = (0, 0, 0) 
red = (255, 0, 0) 

# assigning values to X and Y variable 
X = 700
Y = 700

# create the display surface object 
# of specific dimension..e(X,Y). 
display_surface = pygame.display.set_mode((X, Y )) 

# set the pygame window name 
pygame.display.set_caption('Map') 

# completely fill the surface object 
# with white colour 
display_surface.fill(white) 

obs_1 = [(0, 0), (100, 0), (100, 100), (0, 100)]
obs_2 = [(200, 100), (400, 100), (400, 200), (300, 200), (300, 300), (200, 300)]
obs_3 = [(500, 0), (600, 0), (600, 300), (500, 300)]
obs_4 = [(0, 300), (100, 300), (100, 500), (300, 500), (300, 600), (0, 600)]
obs_5 = [(400, 400), (500, 400), (500, 500), (600, 500), (600, 600), (400, 600)]
obs_6 = [(600, 500), (700, 500), (700, 700), (600, 700)]

obs_list = [obs_1, obs_2, obs_3, obs_4, obs_5, obs_6]

# draw a polygon using draw.polygon() 
# method of pygame. 
# pygame.draw.polygon(surface, color, pointlist, thickness) 
# thickness of line parameter is optional. 
pygame.draw.polygon(display_surface, black, obs_1) 
					
# draw a line using draw.line() 
# method of pygame. 
# pygame.draw.line(surface, color, 
# start point, end point, thickness) 
pygame.draw.polygon(display_surface, black, obs_2) 

pygame.draw.polygon(display_surface, black, obs_3)

pygame.draw.polygon(display_surface, black, obs_4)

pygame.draw.polygon(display_surface, black, obs_5)

pygame.draw.polygon(display_surface, black, obs_6)

# draw a circle using draw.circle() 
# method of pygame. 
# pygame.draw.circle(surface, color, 
# center point, radius, thickness) 

goal = (350, 350)
start = (150, 100)

pygame.draw.circle(display_surface, green, goal, 10, 0)
pygame.draw.circle(display_surface, blue, start, 10, 0)


# draw a ellipse using draw.ellipse() 
# method of pygame. 
# pygame.draw.ellipse(surface, color, 
# bounding rectangle, thickness) 


# draw a rectangle using draw.rect() 
# method of pygame. 
# pygame.draw.rect(surface, color, 
# rectangle tuple, thickness) 
# thickness of line parameter is optional. 

def equation(a, b):
	if a[0] == b[0]:
		return [0, 1, -a[0]]
	else:
		m = (b[1] - a[1]) / (b[0] - a[0])
		c = b[1] - m * b[0]
		return [m, -1, c]

def check(current, destination, obs_list):
	for i in obs_list:
		for j in range(len(i)):
			b = i[j]
			if j == len(i) - 1:
				b_next = j[0]
			else:
				b_next = i[j + 1]
			m, m_y, c = equation(current,destination)
			d_1 = m * b[0] + m_y * b[1] + c
			d_2 = m * b_next[0] + m_y * b_next[1] + c
			if (d_1 <= 0 and d_2 >=0) or (d_2 <= 0 and d_1 >=0):
				return 0
			else:
				return 1

def draw_line(current, destination):
	pygame.draw.line(display_surface, red, [current, destination])
	pygame.draw.circle(display_surface, red, destination, 5, 0)
	pygame.display.flip()


# infinite loop 
while True : 
	
	# iterate over the list of Event objects 
	# that was returned by pygame.event.get() method. 
	for event in pygame.event.get() : 

		# if event object type is QUIT 
		# then quitting the pygame 
		# and program both. 
		if event.type == pygame.QUIT : 

			# deactivates the pygame library 
			pygame.quit() 

			# quit the program. 
			quit() 

		# Draws the surface object to the screen. 
		pygame.display.update() 
