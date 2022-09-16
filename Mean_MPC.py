#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import casadi
import forcespro
import forcespro.nlp
import time
from matplotlib.animation import FuncAnimation, writers
import GPy
import warnings
from scipy import special



class MPC:
	def __init__(self, path, with_obs, L, Ts, N, xmax, xmin, umax, umin, Q_w, Qf_w, R_w, method = None):
		self.path = path
		self.nx = 4
		self.ny = 2
		self.nxi = 3
		self.nu = 2
		self.Ts = Ts
		self.xmax = xmax
		self.xmin = xmin
		self.umax = umax
		self.umin = umin
		self.Q_w = Q_w
		self.Qf_w = Qf_w
		self.R_w = R_w
		self.J = []
		self.X_mean = []
		self.obs_x, self.obs_u, self.obs_x_obs = [], [], []
		self.u_old = np.zeros(self.nu)
		self.N = N
		self.X_pred = []
		self.X_pred_obs = []
		self.goals = []
		self.U = []
		self.L = L
		self.l_r = self.L/2
		self.P_pred = []
		self.obstacles = []
		self.obs_box = []
		self.car_box = []
		self.pred = []
		self.dist = []
		self.time = []
		self.with_obs = with_obs
		self.alpha = 1e-2
		self.beta = 2
		self.kappa_x = 0
		self.kappa_xi = 3 - self.nxi
		self.kappa_l = 3 - self.ny - self.ny
		self.lambda_x = self.alpha**2*(self.nx + self.kappa_x) - self.nx
		self.lambda_xi = self.alpha**2*(self.nxi + self.kappa_xi) - self.nxi
		self.lambda_l = self.alpha**2*(self.ny + self.ny + self.kappa_l) - self.ny - self.ny
		self.gamma_x = np.sqrt(self.lambda_x + self.nx)
		self.gamma_xi = np.sqrt(self.lambda_xi + self.nxi)
		self.gamma_l = np.sqrt(self.lambda_l + self.ny + self.ny)
		self.w_mx = np.array([[self.lambda_x/(self.nx + self.lambda_x)] +
							       [0.5/(self.nx + self.lambda_x)]*2*self.nx]).T
		self.w_cx = np.array([[self.lambda_x/(self.nx + self.lambda_x) + (1 - self.alpha**2 + self.beta)] +
							       [0.5/(self.nx + self.lambda_x)]*2*self.nx]).T
		self.w_mxi = np.array([[self.lambda_xi/(self.nxi + self.lambda_xi)] +
							       [0.5/(self.nxi + self.lambda_xi)]*2*self.nxi]).T
		self.w_cxi = np.array([[self.lambda_xi/(self.nxi + self.lambda_xi) + (1 - self.alpha**2 + self.beta)] +
							       [0.5/(self.nxi + self.lambda_xi)]*2*self.nxi]).T
		self.w_ml = np.array([[self.lambda_l/(self.ny + self.ny + self.lambda_l)] +
							       [0.5/(self.ny + self.ny + self.lambda_l)]*2*(self.ny + self.ny)]).T
		self.w_cl = np.array([[self.lambda_l/(self.nx + self.ny + self.lambda_l) + (1 - self.alpha**2 + self.beta)] +
							       [0.5/(self.ny + self.ny + self.lambda_l)]*2*(self.ny + self.ny)]).T
		self.max_data = 50
		self.nvar = 2*self.nu + self.nx + self.nx*self.nx
		self.npar = self.nx + 2*self.nx + self.nu + self.max_data*self.max_data*self.nx + self.max_data*self.nx + self.nx + self.nx + self.max_data*(self.nx+self.nu) + 1 + self.ny + self.ny*self.ny
					#goal   + Q, Qf     + R       + L                                   + alpha                 + var     + length  + X_train               +  rad + obs_mean + obs_cov12

		self.delta = 0.01
		self.round_upto = 10

		try:
			self.solver = forcespro.nlp.Solver.from_directory("./Mean_MPC")
		except:
			print('Creating MPC')
			self.create_MPC()
		self.solver_backup = forcespro.nlp.Solver.from_directory("./Vanilla_MPC")

	def update_gp(self, obs, u_prev, t):
		u = np.zeros(self.nu)

		if t==1:
			self.kernel = [GPy.kern.RBF(input_dim=self.nx+self.nu, variance=1., lengthscale=1.) for j in range(self.nx)]

		if t==1 or self.max_data+1 > len(self.obs_u):
			if t!=1:
				self.obs_u[-1] = u_prev
			self.obs_x.append(obs)
			self.obs_u.append(u)
			return True
		else:
			self.obs_u[-1] = u_prev
			self.obs_x.append(obs)
			self.obs_u.append(u)
			self.GP_X = np.hstack([self.obs_x[-self.max_data-1:-1], self.obs_u[-self.max_data-1:-1]])
			self.GP_true = self.true_dynamics(self.GP_X[:,:self.nx], self.GP_X[:,self.nx:])
			self.GP_Y = np.array(self.obs_x[-self.max_data:]) - self.GP_true + np.random.normal(scale=0.01, size=(self.max_data, self.nx))

			if self.max_data+1 == len(self.obs_x)-1:
				self.gp = [GPy.models.GPRegression(self.GP_X, np.array([self.GP_Y[:,j]]).T, self.kernel[j]) for j in range(self.nx)]
				with warnings.catch_warnings(record=True) as w:
					warnings.simplefilter("always")
					for j in range(self.nx):
						self.gp[j].optimize(messages=False)
						self.gp[j].optimize_restarts(num_restarts=5, verbose=False)
			else:
				for j in range(self.nx):
					self.gp[j].set_XY(self.GP_X, np.array([self.GP_Y[:,j]]).T)

			gp_L = [np.round(np.linalg.inv(self.gp[i].posterior._woodbury_chol.T), self.round_upto) for i in range(self.nx)]
			gp_alpha = [np.round(self.gp[i].posterior.woodbury_vector[:,0], self.round_upto) for i in range(self.nx)]
			gp_var = [np.round(self.gp[i].parameters[0][0], self.round_upto) for i in range(self.nx)]
			gp_length_inv = [np.round(1/(self.gp[i].parameters[0][1])**2, self.round_upto) for i in range(self.nx)]
			return np.array(gp_L).T, np.array(gp_alpha).T, np.array(gp_var), np.array(gp_length_inv)

	def update_gp_obs(self, obs, t):
		if t==1:
			self.kernel_obs = [GPy.kern.RBF(input_dim=self.nxi, variance=1., lengthscale=1.) for j in range(self.nxi)]

		if t==1 or self.max_data+1 > len(self.obs_x_obs):
			self.obs_x_obs.append(obs)
			return True
		else:
			self.obs_x_obs.append(obs)
			self.GP_X_obs = np.array(self.obs_x_obs[-self.max_data-1:-1])
			self.GP_true_obs = self.true_dynamics_obs(self.GP_X_obs)
			self.GP_Y_obs = np.array(self.obs_x_obs[-self.max_data:]) - self.GP_true_obs + np.random.normal(scale=0.01, size=(self.max_data, self.nxi))

			if self.max_data+1 == len(self.obs_x_obs)-1:
				self.gp_obs = [GPy.models.GPRegression(self.GP_X_obs, np.array([self.GP_Y_obs[:,j]]).T, self.kernel_obs[j]) for j in range(self.nxi)]
				with warnings.catch_warnings(record=True) as w:
					warnings.simplefilter("always")
					for j in range(self.nxi):
						self.gp_obs[j].optimize(messages=False)
						self.gp_obs[j].optimize_restarts(num_restarts=5, verbose=False)
			else:
				for j in range(self.nxi):
					self.gp_obs[j].set_XY(self.GP_X_obs, np.array([self.GP_Y_obs[:,j]]).T)

			gp_L_obs = [np.round(np.linalg.inv(self.gp_obs[i].posterior._woodbury_chol.T), self.round_upto) for i in range(self.nxi)]
			gp_alpha_obs = [np.round(self.gp_obs[i].posterior.woodbury_vector[:,0], self.round_upto) for i in range(self.nxi)]
			gp_var_obs = [np.round(self.gp_obs[i].parameters[0][0], self.round_upto) for i in range(self.nxi)]
			gp_length_inv_obs = [np.round(1/(self.gp_obs[i].parameters[0][1])**2, self.round_upto) for i in range(self.nxi)]
			return np.array(gp_L_obs).T, np.array(gp_alpha_obs).T, np.array(gp_var_obs), np.array(gp_length_inv_obs)


	def solve_MPC_backup(self, x_mean, x_obs, goal, obs_box_, car_box_, extent_o, extent_r, t):
		start = time.time()
		r_o = np.sqrt(extent_o[0]**2 + extent_o[1]**2)
		r_r = np.sqrt(extent_r[0]**2 + extent_r[1]**2)
		dist = self.loss(x_mean, x_obs, r_o + r_r)

		x0 = np.zeros((self.nvar - self.nx*self.nx, self.N))
		problem = {"x0": x0,
				   "xinit": np.concatenate([self.u_old, x_mean])}

		problem["lb"] = np.tile(self.umin, self.N)
		problem["ub"] = np.tile(self.umax, self.N)
		tiled = np.tile(np.concatenate([self.Q_w, self.R_w, self.Qf_w, x_obs[:self.ny], [r_o + r_r]]),(self.N,1))
		stacked = np.hstack((goal, tiled))

		problem["all_parameters"] = np.reshape(stacked,(stacked.shape[0]*stacked.shape[1]))

		output, exitflag, info = self.solver_backup.solve(problem)
		end = time.time()
		sys.stderr.write("t: {}: Backup - FORCESPRO took {} iterations and {} seconds to solve the problem, flag {}.\n" \
						 .format(t, info.it, info.solvetime, exitflag))

#		temp = np.zeros((self.nvar - self.nx*self.nx, self.N))
#		for i in range(0, self.N):
#			temp[:, i] = output['x{0:0{size}d}'.format(i + 1, size=int(np.floor(np.log10(self.N)) + 1))]

		temp = np.reshape(output["sol"], (self.nvar - self.nx*self.nx,self.N), order='F')

		u = temp[self.nu:2*self.nu, :]

		self.u_old = u[:,1]
		self.time.append(end-start)
		self.dist.append(dist)
		self.J.append(info.pobj)
		self.X_mean.append(x_mean)
		self.obstacles.append(x_obs)
		self.obs_box.append(obs_box_)
		self.car_box.append(car_box_)
		self.X_pred.append(temp[2*self.nu:2*self.nu + self.nx, :])
		self.X_pred_obs.append(np.tile(x_obs, (self.N,1)).T)
		self.U.append(u)
		self.pred.append(temp)
		self.goals.append(goal)
		self.P_pred.append(np.zeros((self.nx, self.nx, self.N)))
		throttle, brake = self.Get_Carla_Throttle_Input(u[0,1])
		steer = self.Get_Carla_Steer_Input(u[1,1])
		print('action: ', u[:, 1], 'state: ', x_mean)

		return [throttle, brake, steer], self.X_pred[-1], self.X_pred_obs[-1], exitflag, False

	def solve_MPC(self, x_mean, x_obs, goal, obs_box_, car_box_, extent_o, extent_r, t):
#		x0 = np.zeros((self.nx + self.nx*self.nx + self.nu, self.N))
		gp_out = self.update_gp(x_mean, self.u_old, t)
		gp_out_obs = self.update_gp_obs(x_obs, t)
		goal = np.array(goal)
		if not isinstance(gp_out, tuple):
			print('waiting for data...')
			return self.solve_MPC_backup(x_mean, x_obs, goal, obs_box_, car_box_, extent_o, extent_r, t)
		else:
			start = time.time()
			gp_L, gp_alpha, gp_var, gp_length_inv = gp_out
			gp_L_obs, gp_alpha_obs, gp_var_obs, gp_length_inv_obs = gp_out_obs

			obs_mean = np.zeros((self.nxi, self.N))
			obs_cov12 = np.zeros((self.nxi, self.nxi, self.N))
			if t==1 or 2*self.max_data+1 > len(self.obs_u):
				for i in range(self.N):
					obs_mean[:,i] = x_obs
			else:
				obs_mean[:,0] = x_obs
				for k in range(self.N-1):
					Sigma_p = self.create_sigma_np(obs_mean[:,k], obs_cov12[:,:,k], self.gamma_x)
					Sigma_new = np.zeros((self.nxi, 2*self.nxi+1))
					Q = np.zeros((self.nxi, self.nxi))

					#compute mean
					for j in range(2*self.nxi+1):
						for i in range(self.nxi):
							Sigma_new[i,j] = obs_mean[i,k] + self.predict(Sigma_p[:,j], None, i, (gp_L_obs, gp_alpha_obs, gp_var_obs, gp_length_inv_obs, self.GP_X_obs), 'mean')

					#compute cov
					for i in range(self.nxi):
						Q[i, i] = self.predict(Sigma_p[:,j], None, i, (gp_L_obs, gp_alpha_obs, gp_var_obs, gp_length_inv_obs, self.GP_X_obs), 'cov')
					mean_, obs_cov12[:,:,k+1] = self.update_dist_np(Sigma_new, Q, self.w_mxi, self.w_cxi)
					obs_mean[:,k+1] = np.array(mean_)[:,0]

			r_o = np.sqrt(extent_o[0]**2 + extent_o[1]**2)
			r_r = np.sqrt(extent_r[0]**2 + extent_r[1]**2)
			dist = self.loss(x_mean, x_obs, r_o + r_r)

			x0 = np.zeros((self.nvar, self.N))
			problem = {"x0": x0,
						"xinit": np.concatenate([self.u_old, x_mean, np.zeros((self.nx*self.nx,))])}

			problem["lb"] = np.tile(self.umin, self.N)
			problem["ub"] = np.tile(self.umax, self.N)
			tiled = np.tile(np.concatenate([self.Q_w, self.R_w, self.Qf_w,
											np.reshape(gp_L, (self.max_data * self.max_data * self.nx,), order='F'),
											np.reshape(gp_alpha, (self.max_data * self.nx, ), order='F'),
											gp_var, gp_length_inv,
											np.reshape(self.GP_X, (self.max_data * (self.nx+self.nu), ), order='F'),
											[r_o + r_r]]),(self.N,1))
			stacked = np.hstack((goal, tiled, obs_mean[:self.ny, :].T, np.reshape(obs_cov12[:self.ny, :self.ny, :], (self.ny*self.ny, self.N), order='F').T))

			problem["all_parameters"] = np.reshape(stacked,(stacked.shape[0]*stacked.shape[1]))
			output, exitflag, info = self.solver.solve(problem)
			end = time.time()
			sys.stderr.write("t: {}: FORCESPRO took {} iterations and {} seconds to solve the problem, flag {}.\n" \
							 .format(t, info.it, info.solvetime, exitflag))

#			temp = np.zeros((self.nvar, self.N))
#			for i in range(0, self.N):
#				temp[:, i] = output['x{0:0{size}d}'.format(i + 1, size=int(np.floor(np.log10(self.N)) + 1))]

			temp = np.reshape(output["sol"], (self.nvar,self.N), order='F')

			u = temp[self.nu:2*self.nu, :]

			self.time.append(end-start)
			self.u_old = u[:,1]
			self.dist.append(dist)
			self.J.append(info.pobj)
			self.X_mean.append(x_mean)
			self.obstacles.append(x_obs)
			self.obs_box.append(obs_box_)
			self.car_box.append(car_box_)
			self.X_pred.append(temp[2*self.nu:2*self.nu + self.nx, :])
			self.X_pred_obs.append(obs_mean)
			self.pred.append(temp)
			self.U.append(u)
			self.goals.append(goal)
			self.P_pred.append(np.reshape(temp[2*self.nu + self.nx:2*self.nu + self.nx + self.nx*self.nx, :],(self.nx, self.nx, self.N)))
			throttle, brake = self.Get_Carla_Throttle_Input(u[0,1])
			steer = self.Get_Carla_Steer_Input(u[1,1])

			return [throttle, brake, steer], self.X_pred[-1], self.X_pred_obs[-1], exitflag, True


	def Get_Carla_Steer_Input(self, steer_angle):
		"""
		Given a steering angle in radians, returns the steering input between [-1,1]
		so that it can be applied to the car in the CARLA simulator.
		Max steering angle = 70 degrees = 1.22 radians
		Ref: https://github.com/carla-simulator/ros-bridge/blob/master/carla_ackermann_control/src/carla_ackermann_control/carla_control_physics.py

		Input:
			steer_angle: steering angle in radians

		Output:
			steer_input: steering input between [-1,1]
		"""

		steer_input = (1/1.22)*steer_angle
		steer_input = np.fmax(np.fmin(steer_input, 1.0), -1.0)

		return steer_input


	###############################################################################
	###############################################################################

	def Get_Carla_Throttle_Input(self, accel):
		"""
		Given an acceleration in m/s^2, returns the throttle input between [0,1]
		so that it can be applied to the car in the CARLA simulator.
		Max acceleration = 3.0 m/s^2
		Ref: https://github.com/carla-simulator/ros-bridge/blob/master/carla_ackermann_control/src/carla_ackermann_control/carla_control_physics.py

		Input:
			accel: steering angle in radians

		Output:
			throttle_input: steering input between [0,1]
		"""
		if accel > 0:
			brake_input    = 0.0
			throttle_input = (1/3)*accel
			throttle_input = np.fmax(np.fmin(throttle_input, 1.0), 0.0)
		else:
			throttle_input = 0
			brake_input = (1/3)*accel
			brake_input = np.fmax(np.fmin(throttle_input, 1.0), 0.0)

		return throttle_input, brake_input


	def true_dynamics(self, x, u):
			x = x.T
			u = u.T
			x_new = np.zeros(x.shape)
			beta = np.arctan(self.l_r * np.tan(u[1])/ self.L)
			x_new[0] = x[0] + self.Ts*x[3]*np.cos(x[2] + beta)
			x_new[1] = x[1] + self.Ts*x[3]*np.sin(x[2] + beta)
			x_new[2] = x[2] + self.Ts*np.tan(u[1])*np.cos(beta)*x[3]/self.L
			x_new[3] = x[3] + self.Ts*u[0]
			return x_new.T

	def true_dynamics_obs(self, x):
			return x

	def app_dynamics_obs(self, x):
			return x

	def app_dynamics(self, x, u):
			x_new = casadi.SX.sym('x', self.nx)
			beta = casadi.atan(self.l_r * casadi.tan(u[1])/ self.L)
			x_new[0] = x[0] + self.Ts*x[3]*casadi.cos(x[2] + beta)
			x_new[1] = x[1] + self.Ts*x[3]*casadi.sin(x[2] + beta)
			x_new[2] = x[2] + self.Ts*casadi.tan(u[1])*casadi.cos(beta)*x[3]/self.L
			x_new[3] = x[3] + self.Ts*u[0]
			return x_new


	def predict(self, x, u, i, params, type_):

		gp_L_, gp_alpha, gp_var, gp_length_inv, X_train = params
		if gp_L_.shape[0] == self.max_data*self.max_data:
			gp_L = casadi.reshape(gp_L_[:, i], self.max_data, self.max_data)
		else:
			gp_L = gp_L_[:,:,i]

		if u is not None:
			x_in = casadi.vertcat(x, u)
		else:
			x_in = x[..., np.newaxis]
		r = casadi.sum1(gp_length_inv[i] * (X_train.T - x_in)**2)
		k = gp_var[i] * casadi.exp(-0.5*r).T

		if type_ == 'mean':
			out = k.T @ gp_alpha[:, i]
		elif type_ == 'cov':
			v = gp_L @ k
			out = gp_var[i] - v.T @ v
		return out

	def update_dist(self, X_s_bar, Q, w_m, w_c):
		n = X_s_bar.shape[0]
		mu = casadi.sum1(w_m * X_s_bar.T).T

		slack = 1e-5*casadi.SX.eye(n)
		Sigma = casadi.SX.zeros(n, n)

		for i in range(2*n+1):
			Sigma = Sigma + w_c[i]*(X_s_bar[:,i]-mu) @ (X_s_bar[:,i]-mu).T

		return mu, casadi.chol(Sigma + Q + slack)

	def update_dist_np(self, X_s_bar, Q, w_m, w_c):
		n = X_s_bar.shape[0]
		mu = casadi.sum1(w_m * X_s_bar.T).T

		Sigma = np.zeros((n, n))

		for i in range(2*n+1):
			Sigma = Sigma + w_c[i]*(X_s_bar[:,i]-mu) @ (X_s_bar[:,i]-mu).T

		return mu, np.linalg.cholesky(Sigma + Q + 1e-5*np.eye(n)).T

	def create_sigma(self, mu, Sigma12, gamma):
		n = mu.shape[0]
#
#		X_s = [mu]
#		X_s = X_s + [mu + gamma*Sigma12[:,i] for i in range(n)]
#		X_s = X_s + [mu - gamma*Sigma12[:,i] for i in range(n)]
#		return np.array(X_s)
		X_s = casadi.SX.zeros((n, 2*n+1))

		X_s[:,0] = mu
		for i in range(n):
			X_s[:,i+1] = mu + gamma*Sigma12[:,i]
			X_s[:,i+n+1] = mu - gamma*Sigma12[:,i]
		return X_s

	def create_sigma_np(self, mu, Sigma12, gamma):
		n = mu.shape[0]
		X_s = np.zeros((n, 2*n+1))

		X_s[:,0] = mu
		for i in range(n):
			X_s[:,i+1] = mu + gamma*Sigma12[:,i]
			X_s[:,i+n+1] = mu - gamma*Sigma12[:,i]
		return X_s


	def dynamics(self, z, p):
		x = z[2*self.nu:2*self.nu+self.nx]
		P = casadi.reshape(z[2*self.nu+self.nx:2*self.nu+self.nx+self.nx*self.nx], (self.nx, self.nx))
		deltau = z[:self.nu]
		u = z[self.nu:2*self.nu]
		u_new = deltau + u

		gp_L_ = casadi.reshape(p[self.nx + 2*self.nx + self.nu:self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx], self.max_data*self.max_data, self.nx)
		gp_alpha = casadi.reshape(p[self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx:self.nx + 2*self.nx + self.nu + self.max_data*self.max_data*self.nx + self.max_data*self.nx], self.max_data, self.nx)
		gp_var = p[self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx+self.max_data*self.nx:self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx + self.max_data*self.nx + self.nx]
		gp_length_inv = p[self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx + self.max_data*self.nx + self.nx:self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx + self.max_data*self.nx + self.nx + self.nx]
		X_train = casadi.reshape(p[self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx + self.max_data*self.nx + self.nx + self.nx:self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx + self.max_data*self.nx + self.nx + self.nx + self.max_data*(self.nx+self.nu)], self.max_data, self.nx+self.nu)
		Sigma_p = self.create_sigma(x, P, self.gamma_x)
		Sigma_new = casadi.SX.sym('Sigma_new', self.nx, 2*self.nx+1)
		Q = casadi.SX.zeros(self.nx, self.nx)
		x_temp = casadi.SX.sym('x_temp', self.nx, 2*self.nx+1)

		#compute mean
		for j in range(2*self.nx+1):
			x_temp[:,j] = self.app_dynamics(Sigma_p[:,j], u_new)
			for i in range(self.nx):
				Sigma_new[i,j] = x_temp[i, j] + self.predict(Sigma_p[:,j], u_new, i, (gp_L_, gp_alpha, gp_var, gp_length_inv, X_train), 'mean')

		#compute cov
		for i in range(self.nx):
			Q[i, i] = self.predict(x, u_new, i, (gp_L_, gp_alpha, gp_var, gp_length_inv, X_train), 'cov')
		x_new, P_new = self.update_dist(Sigma_new, Q, self.w_mx, self.w_cx)

		return casadi.vertcat(u_new, x_new, casadi.reshape(P_new, (self.nx*self.nx,1)))

	def objective(self, z, p):
		x = z[2*self.nu:2*self.nu+self.nx]
		P = casadi.reshape(z[2*self.nu+self.nx:2*self.nu+self.nx+self.nx*self.nx], (self.nx, self.nx))
		deltau = z[:self.nu]
		goal = p[:self.nx]
		Q = casadi.diag(p[self.nx:2*self.nx])
		R = casadi.diag(p[2*self.nx: 2*self.nx+self.nu])
		obj = casadi.dot((Q @ (x[:self.nx] - goal)), x[:self.nx] - goal) + casadi.dot((R @ deltau), deltau) + casadi.trace(Q @ P @ P.T)
		return obj

	def objectiveN(self, z, p):
		x = z[2*self.nu:2*self.nu+self.nx]
		P = casadi.reshape(z[2*self.nu+self.nx:2*self.nu+self.nx+self.nx*self.nx], (self.nx, self.nx))
		goal = p[:self.nx]
		Qf = casadi.diag(p[2*self.nx+self.nu: 3*self.nx+self.nu])
		obj = casadi.dot((Qf @ (x[:self.nx] - goal)), x[:self.nx] - goal) + casadi.trace(Qf @ P @ P.T)
		return obj


	def inequality(self, z,p):
		r = p[self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx + self.max_data*self.nx + self.nx + self.nx + self.max_data*(self.nx+self.nu)]
		x = z[2*self.nu:2*self.nu+self.ny]
		x_o = p[self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx + self.max_data*self.nx + self.nx + self.nx + self.max_data*(self.nx+self.nu) + 4 : self.nx + 2*self.nx + self.nu+self.max_data*self.max_data*self.nx + self.max_data*self.nx + self.nx + self.nx + self.max_data*(self.nx+self.nu) + 4 + self.ny]

		return [self.loss(x, x_o, r) - self.delta]

	def loss(self, x, x_o, r):
		return r**2 - (x_o[:self.ny] - x[:self.ny]).T @ (x_o[:self.ny] - x[:self.ny])


	def create_MPC(self):
		model = forcespro.nlp.SymbolicModel(self.N)
		model.N = self.N
		model.nvar = self.nvar
		model.neq = self.nu + self.nx + self.nx*self.nx
		model.npar = self.npar
		model.nh = 1
		model.ineq = self.inequality
		model.hu = [0]
		model.objective = self.objective
		model.objectiveN = self.objectiveN
		model.eq = self.dynamics
		model.E = np.concatenate([np.zeros((self.nu + self.nx + self.nx*self.nx, self.nu)), np.eye(self.nu + self.nx + self.nx*self.nx)], axis=1)
		model.ubidx = range(self.nu-1, 2*self.nu)
		model.lbidx = range(self.nu-1, 2*self.nu)
		model.xinitidx = range(self.nu, self.nvar)
		codeoptions = forcespro.CodeOptions('Mean_MPC')
		codeoptions.maxit = 2000
		codeoptions.printlevel = 1
		codeoptions.optlevel = 1
		codeoptions.noVariableElimination = 1.
		codeoptions.nlp.stack_parambounds = True
		codeoptions.overwrite = 1
		codeoptions.forcenonconvex = 1
		codeoptions.nlp.linear_solver = 'symm_indefinite_fast'
		
		output = ("sol", [], [])
		self.solver = model.generate_solver(codeoptions, [output])

	def save(self):
		print('cost:', np.sum(self.J))
		print('max loss:', np.max(self.dist))
		print('max time:', np.max(self.time))
		print('mean (std) time: {} ({})'.format(np.mean(self.time), np.std(self.time)))
		np.save(self.path + 'X_pred.npy', np.array(self.X_pred))
		np.save(self.path + 'X_pred_obs.npy', np.array(self.X_pred_obs))
		np.save(self.path + 'X_mean.npy', np.array(self.X_mean))
		np.save(self.path + 'J.npy', np.array(self.J))
		np.save(self.path + 'U.npy', np.array(self.U))
		np.save(self.path + 'dist.npy', np.array(self.dist))
		np.save(self.path + 'time.npy', np.array(self.time))




	def render(self, world):
		T = len(self.X_mean)
		self.X_mean = np.array(self.X_mean)
		plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
		writer = writers['ffmpeg'](fps=5)
		fig = plt.figure(figsize = (10,8))
		ax = fig.gca()
		x = []
		y = []

		w_list = world.map.generate_waypoints(2.0)
		waypoints = np.array([[w.transform.location.x, w.transform.location.y]  for w in w_list])
		ax.scatter(waypoints[:,0], waypoints[:,1], color='tab:gray', marker='.', s=1)
		spawn_points = [[w.location.x, w.location.y] for w in world.map.get_spawn_points()]
		spawn_points = np.array(spawn_points)
		ax.scatter(spawn_points[:,0], spawn_points[:,1], color='r', marker='.')
		for i in range(spawn_points.shape[0]):
			plt.annotate('{}'.format(i), tuple(spawn_points[i,:]))
			
			
		line, = ax.plot(x, y,
					label='Robot Trajectory',
					color='k')
		line2, = ax.plot(x, y, 'r--', label='State Prediction')
		line3, = ax.plot(x, y, 'p--', label='Obstacle Prediction')
		goal_pl, = ax.plot(x, y, color='tab:blue')
		line_r = ax.scatter(x, y, label='Car State', marker='.', color = 'tab:blue')
		line_o = ax.scatter(x, y, label='Obstacle State', marker='.', color = 'tab:green')
		line_oo, = ax.plot(x, y, label='Obstacle Box', color = 'tab:green')
		line_rr, = ax.plot(x, y, label='Car Box', color = 'tab:blue')
		timestep = ax.text(-5.5, -5.7, '', fontsize=15)
		def func(t):
			x_traj = self.X_mean[:t,0]
			y_traj = self.X_mean[:t,1]

			timestep.set_text('t = {}'.format(t))
			line_r.set_offsets(self.X_mean[t,:2])
			line_o.set_offsets(self.obstacles[t][:2])

			line.set_data(x_traj, y_traj)
			line2.set_data(self.X_pred[t][0, :], self.X_pred[t][1, :])
			line3.set_data(self.X_pred_obs[t][0, :], self.X_pred_obs[t][1, :])
			if self.with_obs:
				line_oo.set_data(self.obs_box[t][:,0], self.obs_box[t][:,1])
			line_rr.set_data(self.car_box[t][:,0], self.car_box[t][:,1])
			goal_pl.set_data(self.goals[t][:,0], self.goals[t][:,1])

		# Animate
		ani = FuncAnimation(fig=fig, func=func, frames=T)
		plt.grid()

		plt.scatter(self.X_mean[0,0], self.X_mean[0,1], color='tab:blue')
		plt.annotate('starting position', tuple(self.X_mean[0,:2]))
		ax.axis([-140, 50, -80, 50])
		plt.legend()

		ani.save(self.path + 'vid_{}.mp4'.format(time.strftime("%m%d-%H%M%S")), writer=writer)	
		plt.close('all')
		print('---------Done!-------')

