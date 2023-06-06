import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import Axes3D
import scipy
#import pandas as pd
import time
import data


# define a new tf.keras.layers.Layer per recommendation of 
# TensorFlow workflow (instead of using tf.keras.layers.Lambda)
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale_function = lambda x : x):
        super(ScaleLayer, self).__init__()
        self.scale = scale_function

    def call(self, inputs):
        return self.scale(inputs)


class PINN_FisherKPP:
    def __init__(self, data, hidden_layers, learning_rate = 0.01, lr_decay = 'piecewise', activation = None):
        t_0, x_0, u_0, t_b, x_b, u_b, t_r, x_r, bounds = data
        self.t_min, self.t_max, self.x_min, self.x_max = bounds

        self.x_0 = tf.convert_to_tensor(x_0, dtype = tf.float32)
        self.u_0 = tf.convert_to_tensor(u_0, dtype = tf.float32)
        self.t_0 = tf.convert_to_tensor(t_0, dtype = tf.float32)
        self.x_b = tf.convert_to_tensor(x_b, dtype = tf.float32)
        self.u_b = tf.convert_to_tensor(u_b, dtype = tf.float32)
        self.t_b = tf.convert_to_tensor(t_b, dtype = tf.float32)
        self.x_r = tf.convert_to_tensor(x_r, dtype = tf.float32)
        self.t_r = tf.convert_to_tensor(t_r, dtype = tf.float32)

        
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate

    def residual(self, t, x, u, u_t, u_x, u_xx):
        return u_t - u_xx - u * (1 - u)


    def build_model(self):

        model = tf.keras.Sequential()
        # input layer (t, x)
        model.add(tf.keras.layers.InputLayer(2))
        # scale inputs within bounds
        scale_function = lambda inputs : 2 * (inputs - tf.convert_to_tensor([self.t_min, self.x_min])) / tf.convert_to_tensor([self.t_max - self.t_min, self.x_max - self.x_min]) - 1.0
        model.add(ScaleLayer(scale_function = scale_function))

        # hidden layers
        for layer in self.hidden_layers:
            # activation function - tanh
            # Xavier normal initializer
            model.add(tf.keras.layers.Dense(layer, 
                activation = 'tanh', 
                kernel_initializer = 'glorot_normal'))

        # output layer with sigmoid activation
        model.add(tf.keras.layers.Dense(1, activation = self.activation))

        return model


    def evaluate_residual(self, model):

        with tf.GradientTape(persistent = True) as tape:
            t = self.t_r
            x = self.x_r
            tape.watch(t)
            tape.watch(x)
            u_r = model(tf.stack([t, x], axis = 1))
            u_x = tape.gradient(u_r, x)
        #u_x = tape.gradient(u_r, x)
        u_t = tape.gradient(u_r, t)
        u_xx = tape.gradient(u_x, x)

        del tape

        return self.residual(t, x, u_r, u_t, u_x, u_xx)

    def compute_loss(self, model, scale_initial_loss = None):
        loss = 0
        
        u_0_pred = model(tf.stack([self.t_0, self.x_0], axis = 1))
        scale_initial_by = 1 if scale_initial_loss is None else scale_initial_loss
        loss += scale_initial_by * tf.reduce_mean(tf.square(self.u_0 - u_0_pred))

        u_b_pred = model(tf.stack([self.t_b, self.x_b], axis = 1))
        loss += tf.reduce_mean(tf.square(self.u_b - u_b_pred))

        res = self.evaluate_residual(model)
        loss += tf.reduce_mean(tf.square(res))

        return loss
    #@tf.function
    def compute_grad(self, model):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            loss = self.compute_loss(model)
        grad = tape.gradient(loss, model.trainable_variables)
        del tape
        return loss, grad



    def train(self, num_epochs):
        model = self.build_model()

        learning_rate_fn = self.learning_rate

        if self.lr_decay == 'piecewise':
            learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                [num_epochs//3, 2*num_epochs//3], 
                [self.learning_rate, self.learning_rate / 10, self.learning_rate / 20])

        if self.lr_decay == 'exponential':
            learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps = 100000, decay_rate = 0.96, staircase = True)

        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_fn)
        losses = []

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            loss, grad = self.compute_grad(model)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            losses.append(loss.numpy())
            if epoch % 10 == 0:
                print("Epoch {}: loss = {}, time elapsed {}".format(
                    epoch, loss.numpy(), time.time() - start_time))

        return model, losses

    def plot_dataset(self):
        pass


    def plot_loss(self, losses):
        fig = plt.figure()
        hist = fig.add_subplot(111)
        hist.plot(range(0, len(losses)), losses)
        hist.set_xlabel('# of epochs')
        hist.set_ylabel('loss')
        plt.savefig('loss_plot.png', bbox_inches = 'tight')

    # this plot function is based on existing pyplot tutorials from Github 
    def plot_prediction(self, model):
        # Set up meshgrid
        N = 600
        t_dom = np.linspace(self.t_min, self.t_max, N + 1)
        x_dom = np.linspace(self.x_min, self.x_max, N + 1)
        t_grid, x_grid = np.meshgrid(t_dom, x_dom)
        input_data = np.vstack([t_grid.flatten(),x_grid.flatten()]).T

        # Determine predictions of u(t, x)
        u_pred = model(tf.cast(input_data, tf.float32)).numpy()

        # Reshape upred
        u_pred = u_pred.reshape(N+1,N+1)

        # Surface plot of solution u(t,x)
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(t_grid, x_grid, u_pred, cmap='viridis');
        ax.view_init(35,35)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_zlabel('$u_\\theta(t,x)$')
        ax.set_title('Solution of FKPP equation');
        plt.savefig('FKPP_Solution.pdf', bbox_inches='tight', dpi=300)

    def train_adaptive(self, num_epochs):
        pass

    def train_and_show_results(self, num_epochs = 100, adaptive = False):
        if not adaptive: 
            model, losses = self.train(num_epochs)
            self.plot_loss(losses)
            self.plot_prediction(model)
        if adaptive:
            pass



if __name__ == "__main__":
    hidden_layers = [20, 20, 20, 20, 20, 20, 20, 20]
    data = data.generate_data(data.u_0, data.u_b)
    my_PINN = PINN_FisherKPP(data, hidden_layers)
    my_PINN.train_and_show_results()









