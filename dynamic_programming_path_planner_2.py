# Implementation of:
#
#
#
#
#
# Author: Evan Krell

#from _future_ import print_function
import numpy as np
import math

# Destination position
G = (7, 7)

# Environment env: Occupancy grid
# env[i][j] = 0 if obstacle, env[i][j] = 1 if free
# Size (15, 15)
env = np.array([
      [ 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0 ],
      [ 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1 ],
      [ 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1 ],
      [ 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    ])

env = np.array([
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    ])



# environment size
m = len(env)
n = len(env[0])
M = m * n

# Minimum distance between neighbors
d_min = 0
# Maximum distance between neighbors
d_max = math.sqrt(2)
# Maximum cost
D_max = M * d_max  # D_max must be > (M - 1) * d_max


# Get neighbors
def getNeighbors(i, m, n):

    B = []

    # Up
    if(i[0] - 1 >= 0):
        B.append((i[0] - 1, i[1], "^"))
    # Down
    if(i[0] + 1 < m):
        B.append((i[0] + 1, i[1], "v"))
    # Left
    if(i[1] - 1 >= 0):
        B.append((i[0], i[1] - 1, "<"))
    # Right
    if(i[1] + 1 < n):
        B.append((i[0], i[1] + 1, ">"))
    # Up-Left
    if(i[0] - 1 >= 0 and i[1] - 1 >= 0 ):
        B.append((i[0] - 1, i[1] - 1, "a"))
    # Up-Right
    if(i[0] - 1 >= 0 and i[1] + 1 < n):
        B.append((i[0] - 1, i[1] + 1, "b"))
    # Down-Left
    if(i[0] + 1 < m and i[1] - 1 >= 0):
        B.append((i[0] + 1, i[1] - 1, "c"))
    # Down-Right
    if(i[0] + 1 < m and i[1] + 1 < n):
        B.append((i[0] + 1, i[1] + 1, "d"))

    return B


# Distance between location i and j
def distance(i, j):
    dist = pow((pow(i[0] - j[0], 2) + pow(i[1] - j[1], 2)), 0.5)
    return dist


def f(i, m, n, cost2go, D_max):

    B = getNeighbors(i, m, n)

    cost_min = D_max
    b_min = (None, None, "-")
    for b in B:
        cost = distance(i, b) + cost2go[b[0]][b[1]]
        if cost < cost_min:
            cost_min = cost
            b_min = b

    return cost_min, b_min


def getNewCost(i, G, env, m, n, cost2go, D_max):

    cost = D_max
    action = " "

    # Check if target location
    if i == G:
        cost = 0

    # Check if obstacle
    elif env[i[0]][i[1]] == 0:
        cost = D_max

    # Else, calculate cost normally
    else:
        c, b = f(i, m, n, cost2go, D_max)
        cost = min(D_max, c )
        action = b[2]

    return cost, action

cost2go = [[D_max for j in range(n)] for i in range(m)]
action2go = [["-" for j in range(n)] for i in range(m)]

for k in range(M*M):
    for row in range(m):
        for col in range(n):
            i = (row, col)
            cost2go[i[0]][i[1]], action2go[i[0]][i[1]] = getNewCost(i, G, env, m, n, cost2go, D_max)

print (np.array(cost2go))
print(np.array(action2go))



