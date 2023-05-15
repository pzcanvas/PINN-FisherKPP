import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

# data type
DTYPE = 'float32'

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


def generate_data(num_0: int = N_INITIAL, num_b: int = N_BOUNDARY, num_r: int = N_RESIDUAL, t_min = T_MIN, t_max = T_MAX, x_min = X_MIN, x_max = X_MAX):

	#tf.keras.backend.set_floatx(DTYPE)

	#t_lb = tf.constant(t_min, dtype = DTYPE)
	#t_ub = tf.constant(t_max, dtype = DTYPE)
	#x_lb = tf.constant(x_min, dtype = DTYPE)
	#x_ub = tf.constant(x_max, dtype = DTYPE)


	# sampling initial boundary points
	t_0 = np.ones((num_0, 1)) * t_min
	x_0 = np.random.uniform(x_min, x_max, num_0)
	u_0 = u_0(x_0)

	# sampling boundary points
	t_b = np.random.uniform(t_min, t_max, num_b)
	x_b = np.random.choice([x_min, x_max], size = num_b)
	u_b = u_b(t_b, x_b)

	# sampling collocation points
	t_r = np.random.uniform(t_min, t_max, num_r)
	x_r = np.random.uniform(x_min, x_max, num_r)





