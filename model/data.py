import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread


# number of sampled points 
N_INITIAL = 50
N_BOUNDARY = 50
N_RESIDUAL = 10000

# set boundary
T_MIN = 0.
T_MAX = 0.1
X_MIN = 0.
X_MAX = 1.

# initial conditions
''' 
'''
def u_0(x):
	return 1 / (1 + np.exp(x)) ** 2

# boundary conditions
def u_b(t, x):
	return 1 / (1 + np.exp(x - 5 * t)) ** 2

def get_bounds(t_min = T_MIN, t_max = T_MAX, x_min = X_MIN, x_max = X_MAX):
	return t_min, t_max, x_min, x_max

def generate_data(u_0, u_b, num_0: int = N_INITIAL, num_b: int = N_BOUNDARY, num_r: int = N_RESIDUAL, bounds = get_bounds()):
	t_min, t_max, x_min, x_max = bounds

	# sampling initial boundary points
	t_0 = np.ones(num_0) * t_min
	x_0 = np.random.uniform(x_min, x_max, num_0)
	u_0 = u_0(x_0)

	# sampling boundary points
	t_b = np.random.uniform(t_min, t_max, num_b)
	x_b = np.random.choice([x_min, x_max], size = num_b)
	u_b = u_b(t_b, x_b)

	# sampling collocation points
	t_r = np.random.uniform(t_min, t_max, num_r)
	x_r = np.random.uniform(x_min, x_max, num_r)

	return t_0, x_0, u_0, t_b, x_b, u_b, t_r, x_r, bounds





