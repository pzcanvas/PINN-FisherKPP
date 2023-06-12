import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import Axes3D
import scipy
import time
import data
import os


# define a new tf.keras.layers.Layer per recommendation of 
# TensorFlow workflow (instead of using tf.keras.layers.Lambda)
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale_function = lambda x : x):
        super(ScaleLayer, self).__init__()
        self.scale = scale_function

    def call(self, inputs):
        return self.scale(inputs)


class PINN_FisherKPP:
    def __init__(self, my_data, hidden_layers, learning_rate = 0.01, lr_decay = None, 
    	num_epochs = 200, adaptive = False, scale_initial_by = 100, add_gelu = False):
        t_0, x_0, u_0, t_b, x_b, u_b, t_r, x_r, bounds = my_data
        self.t_min, self.t_max, self.x_min, self.x_max = bounds

        self.x_0 = tf.convert_to_tensor(x_0, dtype = tf.float32)
        self.u_0 = tf.convert_to_tensor(u_0, dtype = tf.float32)
        self.t_0 = tf.convert_to_tensor(t_0, dtype = tf.float32)
        self.x_b = tf.convert_to_tensor(x_b, dtype = tf.float32)
        self.u_b = tf.convert_to_tensor(u_b, dtype = tf.float32)
        self.t_b = tf.convert_to_tensor(t_b, dtype = tf.float32)
        self.x_r = tf.convert_to_tensor(x_r, dtype = tf.float32)
        self.t_r = tf.convert_to_tensor(t_r, dtype = tf.float32)
        self.bounds = bounds

        
        self.hidden_layers = hidden_layers
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.adaptive = adaptive
        self.scale_initial_by = scale_initial_by
        self.add_gelu = add_gelu

    # compute the strong residual 
    def residual(self, t, x, u, u_t, u_x, u_xx, case):
    	if case == 1:
    		return u_t - u_xx - 6 * u * (1 - u)
    	if case == 2 or case == 3:
    		return u_t - u_xx - u * (1 - u)

    # exact solution
    def u_exact_case1(self, t, x):
        return 1 / ((1 + np.exp(x - 5 * t)) ** 2)
    def u_exact_case2(self, t, x, l = .5):
        return l * np.exp(t) / (1 - l + l * np.exp(t))

    def build_model(self):

        model = tf.keras.Sequential()
        # input layer (t, x)
        model.add(tf.keras.layers.InputLayer(2))
        # scale inputs within bounds
        scale_function = lambda inputs : 2 * (inputs - tf.convert_to_tensor([self.t_min, self.x_min], dtype = tf.float32)) / tf.convert_to_tensor([self.t_max - self.t_min, self.x_max - self.x_min], dtype = tf.float32) - 1.0
        model.add(ScaleLayer(scale_function = scale_function))

        # hidden layers
        for i in range(len(self.hidden_layers) - 1):
            # activation function - tanh
            # Xavier normal initializer
            model.add(tf.keras.layers.Dense(self.hidden_layers[i], 
                activation = 'tanh', 
                kernel_initializer = 'glorot_normal'))
        if self.add_gelu:
        	model.add(tf.keras.layers.Dense(self.hidden_layers[-1], 
                activation = 'gelu', 
                kernel_initializer = 'glorot_normal'))
        else:
        	model.add(tf.keras.layers.Dense(self.hidden_layers[i], 
                activation = 'tanh', 
                kernel_initializer = 'glorot_normal'))

        # output layer with sigmoid activation
        model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

        return model

    def evaluate_residual(self, model, t, x, case = 1):

        with tf.GradientTape(persistent = True) as tape:
            tape.watch(t)
            tape.watch(x)

            u_r = model(tf.stack([t, x], axis = 1))
            u_x = tape.gradient(u_r, x)

        u_t = tape.gradient(u_r, t)
        u_xx = tape.gradient(u_x, x)

        del tape
        res = self.residual(t, x, u_r, u_t, u_x, u_xx, case)
        return res

    def compute_loss(self, model, case = 1):
        loss = 0
        
        u_0_pred = model(tf.stack([self.t_0, self.x_0], axis = 1))
        scale_initial = 1 if self.scale_initial_by is None else self.scale_initial_by
        loss += scale_initial * tf.reduce_mean(tf.square(self.u_0 - u_0_pred))

        u_b_pred = model(tf.stack([self.t_b, self.x_b], axis = 1))
        loss += tf.reduce_mean(tf.square(self.u_b - u_b_pred))

        res = self.evaluate_residual(model, self.t_r, self.x_r, case)
        loss += tf.reduce_mean(tf.square(res))

        return loss

    def compute_grad(self, model, case = 1):
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(model.trainable_variables)
            loss = self.compute_loss(model, case)
        grad = tape.gradient(loss, model.trainable_variables)
        del tape
        return loss, grad



    def train(self, num_epochs, case = 1):
        model = self.build_model()

        learning_rate_fn = self.learning_rate

        if self.lr_decay == 'piecewise':
            learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
        if self.lr_decay == 'exponential':
            learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps = 100000, decay_rate = 0.96, staircase = True)

        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_fn)
        losses = []

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            loss, grad = self.compute_grad(model, case)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            losses.append(loss.numpy())
            if epoch % 50 == 0:
                print("Epoch {}: loss = {}, time elapsed {}".format(
                    epoch, loss.numpy(), time.time() - start_time))
                if self.adaptive:
                    test_t, test_x = data.sample_collocation(100, self.bounds)
                    test_t = tf.convert_to_tensor(test_t, dtype = tf.float32)
                    test_x = tf.convert_to_tensor(test_x, dtype = tf.float32)
                    res = self.evaluate_residual(model, test_t, test_x, case)
                    idx = tf.math.top_k(tf.reshape(res,[-1]), k = 20).indices
                    new_t = tf.gather(test_t, idx)
                    new_x = tf.gather(test_x, idx)
                    self.t_r = tf.concat([self.t_r, new_t], 0)
                    self.x_r = tf.concat([self.x_r, new_x], 0)


        return model, losses

    def plot_dataset(self):
        pass


    def plot_loss(self, losses, subfolder):
        fig = plt.figure()
        hist = fig.add_subplot(111)
        hist.plot(range(0, len(losses)), losses)
        hist.set_xlabel('# of epochs')
        hist.set_ylabel('loss')
        plt.savefig(subfolder + 'loss_plot.png', bbox_inches = 'tight')

    # this plot function is based on existing pyplot tutorials from Github 
    def plot_prediction(self, model, subfolder):
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
        plt.savefig(subfolder + 'FKPP_Solution.pdf', bbox_inches='tight', dpi=300)

        



    def train_and_show_results(self, subfolder, case = 1):
        model, losses = self.train(self.num_epochs, case)
        self.plot_loss(losses, subfolder)
        self.plot_prediction(model, subfolder)

        # compare errors
        if case == 1:
        	func_u_0 = data.u_0_case1
        	func_u_b = data.u_b_case1
        if case == 2:
        	func_u_0 = data.u_0_case2
        	func_u_b = data.u_b_case2
        if case == 3:
        	func_u_0 = data.u_0_case3
        	func_u_b = data.u_b_case3

        u_approx, u_predict, u_true = self.numerical_approx(model, func_u_0, func_u_b, case = case)
        if case == 1 or case == 2:
            print("L2 error (prediction vs. exact)): {}".format(np.linalg.norm(u_true.reshape(u_predict.shape) - u_predict)))
            print("L2 error (numerical approx vs. exact)): {}".format(np.linalg.norm(u_true - u_approx)))
            print("L-infinity error (prediction vs. exact)): {}".format(np.linalg.norm(u_true.reshape(u_predict.shape) - u_predict, np.inf)))
            print("L-infinity error (numerical approx vs. exact)): {}".format(np.linalg.norm(u_true - u_approx, np.inf)))
            
        
        print("L2 error (numerical approx vs. prediction)): {}".format(np.linalg.norm(u_predict - u_approx.reshape(u_predict.shape))))
        print("L-infinity error (numerical approx vs. prediction)): {}".format(np.linalg.norm(u_predict - u_approx.reshape(u_predict.shape), np.inf)))

        return model, losses



    def numerical_approx(self, model, func_u_0, func_u_b, case):
        N = 10
        M = 100
        tspace = np.linspace(self.t_min, self.t_max, M + 1)
        xspace = np.linspace(self.x_min, self.x_max, N + 1)

        #choose such N,M for stability purposes


        dx=(self.x_max - self.x_min)/N  #dx should be 0.1
        dt=(self.t_max - self.t_min)/M #dt should be 0.001

        r=dt/(dx)**2

        U=np.zeros((N+1,M+1))
        #boundary condition at x=0
        U[0,:]=func_u_b(tspace, np.zeros(tspace.shape))
        #boundary condition at x=1
        U[N,:]=func_u_b(tspace, np.ones(tspace.shape))
        #initial condition
        U[:,0]=func_u_0(xspace)

        #initialize Q,A,V
        Q=np.zeros((N+1,M+1))
        Q[:,0]=1-2*r+6*dt-6*dt*U[:,0]
        A=np.zeros((N-1,N-1))
        #tridagonal matrix 
        test_data = [r*np.ones(N-1),np.zeros(N-1),r*np.ones(N-1)] 
        # list of all the data
        diags = [-1,0,1] 
        A= scipy.sparse.spdiags(test_data,diags,N-1,N-1,format='csc')
        A=A.todense()
        np.fill_diagonal(A , Q[:,0])
        V=U[1:N,0]
        n=0
        #making sure solution stays between 0 and 1 at all times 
        while all(k >=0 and k<1 for k in U[:,n]) and n<M:
            U[1:N,n+1]=np.dot(A,V)
            U[1,n+1]=r*U[0,n]+Q[1,n]*U[1,n]+r*U[2,n]
            U[N-1,n+1]=r*U[N-2,n]+Q[N-1,n]*U[N-1,n]+r*U[N,n]
            Q[:,n+1]=1-2*r+6*dt-6*dt*U[:,n+1]
            np.fill_diagonal(A , Q[:,n+1])
            V=U[1:N,n+1]
            n+=1

        t_grid, x_grid = np.meshgrid(tspace, xspace)
        input_tensor = tf.cast(np.vstack([t_grid.flatten(), x_grid.flatten()]).T, tf.float32)

        u_pred = model(input_tensor)
        u_true = None
        if case == 1:
        	u_true = self.u_exact_case1(t_grid.flatten(), x_grid.flatten())

        	# compare results at the final time
        	for i in range(len(xspace)):
        		x = xspace[i]
        		soln_exact = self.u_exact_case1(self.t_max, x)
        		pinn_input = tf.cast(np.vstack([self.t_max, x]).T, tf.float32)
        		soln_pinn = model(pinn_input)
        		soln_numerical = U[i, -1]
        		print("x value: {},    exact solution: {},    PINN: {},    numerical: {}".format(x, soln_exact, soln_pinn, soln_numerical))
        		print("PINN-abs:  {}, numerical-abs: {}".format(np.abs(soln_exact - soln_pinn), np.abs(soln_exact - soln_numerical)))
        if case == 2:
        	u_true = self.u_exact_case2(t_grid.flatten(), x_grid.flatten())

        

        return (U.flatten(), u_pred, u_true)

  


if __name__ == "__main__":
    my_path = os.path.dirname(__file__)
    hidden_layers = [20 for i in range(10)]
    num_epochs = 5000
    tf.random.set_seed(0)
    
    # Case 1 adaptive, with scaling
    my_data_case1 = data.generate_data(data.u_0_case1, data.u_b_case1)
    my_PINN_case1 = PINN_FisherKPP(my_data_case1, hidden_layers, learning_rate = 0.01, lr_decay = 'piecewise', num_epochs = num_epochs, adaptive = True, scale_initial_by = 100, add_gelu = False)
    case1_path = os.path.join(my_path, 'experiments/case1-as/')
    os.makedirs(case1_path)
    my_PINN_case1.train_and_show_results(case1_path, case = 1)










