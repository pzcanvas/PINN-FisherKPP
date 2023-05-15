import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
import pandas as pd

import time


class PINN:
	def __init__(self, data, bounds, hidden_layers):
		t_0, x_0, u_0, t_b, x_b, u_b, t_r, x_r = data
		self.t_min, self.t_max, self.x_min, self.x_max = bounds

		self.x_0 = tf.Tensor(x_0)
		self.u_0 = tf.Tensor(u_0)
		self.t_0 = tf.Tensor(t_0)
		self.x_b = tf.Tensor(x_b)
		self.u_b = tf.Tensor(u_b)
		self.x_r = tf.Tensor(x_r)
		self.t_r = tf.Tensor(t_r)
		
		self.hidden_layers = hidden_layers

	def residual(t, x, u, u_t, u_x, u_xx):
		return u_t - u_xx - u * (1 - u)

	def build_model(self):

		model = tf.keras.Sequential()
		# input layer (t, x)
		model.add(tf.keras.Input(shape = 2))

		# hidden layers
		for layer in self.hidden_layers:
			# activation function - tanh
			# Xavier normal initializer
			model.add(tf.keras.layers.Dense(layer, 
				activation = 'tanh', 
				kernel_initializer = 'glorot_normal'))

		# output layer
		model.add(tf.keras.layers.Dense(1))

		return model


	def evaluate_residual(self, model):

		with tf.GradientTape(persistent = True) as tape:
			t = self.t_r
			x = self.x_r
			tape.watch(t)
			tape.watch(r)
			u_r = model(tf.stack([t, x], axis = 1))
			u_x = tape.gradient(u_r, x)
		#u_x = tape.gradient(u_r, x)
		u_t = tape.gradient(u_r, t)
		u_xx = tape.gradient(u_x, x)

		del tape

		return residual(t, x, u_r, u_t, u_x, u_xx)

	def compute_loss(self, model):
		loss = 0

		u_0_pred = model(tf.stack([self.t_0, self.x_0], axis = 1))
		loss += tf.reduce_mean(tf.square(self.u_0 - u_0_pred))

		u_b_pred = model(tf.stack([self.t_b, self.x_b], axis = 1))
		loss += tf.reduce_mean(tf.square(self.u_b - u_b_pred))

		res = self.evaluate_residual(model)
		loss += tf.reduce_mean(tf.square(res))

		return loss

	def compute_grad(self, model):
		with tf.GradientTape() as tape:
			tape.watch(model.trainable_variables)
			loss = self.compute_loss(model)
		grad = tape.gradient(loss, model.trainable_variables)
		del tape
		return loss, grad

	def train(self, num_epochs):

		#for epoch in range(num_epochs):




