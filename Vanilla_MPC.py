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


class MPC:
	def __init__(self, path, with_obs, L, Ts, N, xmax, xmin, umax, umin, Q_w, Qf_w, R_w, method = None):
		self.path = path
		self.nx = 4
		self.ny = 2
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
		self.N = N
		self.X_pred = []
		self.X_pred_obs = []
		self.dist = []
		self.time = []
		self.goals = []
		self.U = []
		self.L = L
		self.l_r = self.L/2
		self.obstacles = []
		self.obs_box = []
		self.car_box = []
		self.u_old = np.zeros(self.nu)
		self.with_obs = with_obs

		try:
			self.solver = forcespro.nlp.Solver.from_directory("./Vanilla_MPC")
		except:
			self.create_MPC()

	def solve_MPC(self, x_mean, obs, goal, obs_box_, car_box_, extent_o, extent_r, t):
		goal = np.array(goal)
		start = time.time()
		r_o = np.sqrt(extent_o[0]**2 + extent_o[1]**2)
		r_r = np.sqrt(extent_r[0]**2 + extent_r[1]**2)
		dist = self.loss(x_mean, obs, r_o + r_r)
		x0 = np.zeros((self.nx + 2*self.nu, self.N))
		problem = {"x0": x0,
				   "xinit": np.concatenate([self.u_old, x_mean])}

		problem["lb"] = np.tile(self.umin, self.N)
		problem["ub"] = np.tile(self.umax, self.N)
		tiled = np.tile(np.concatenate([self.Q_w, self.R_w, self.Qf_w, [(r_o + 2*r_r)], obs[:2]]),(self.N,1))
		stacked = np.hstack((goal, tiled))

		problem["all_parameters"] = np.reshape(stacked,(stacked.shape[0]*stacked.shape[1]))

		output, exitflag, info = self.solver.solve(problem)
		end = time.time()
		sys.stderr.write("t: {}: FORCESPRO took {} iterations and {} seconds to solve the problem, flag {}.\n" \
						 .format(t, info.it, info.solvetime, exitflag))

#		temp = np.zeros((2*self.nu + self.nx, self.N))
#		for i in range(0, self.N):
#			temp[:, i] = output['x{0:0{size}d}'.format(i + 1, size=int(np.floor(np.log10(self.N)) + 1))]
		temp = np.reshape(output["sol"], (self.nx+2*self.nu,self.N), order='F')
		u = temp[self.nu:2*self.nu, :]
		self.u_old = u[:,1]
		self.time.append(end-start)
		self.dist.append(dist)
		self.J.append(info.pobj)
		self.X_mean.append(x_mean)
		self.obstacles.append(obs)
		self.obs_box.append(obs_box_)
		self.car_box.append(car_box_)
		self.X_pred.append(temp[2*self.nu:2*self.nu + self.nx, :])
		self.X_pred_obs.append(np.tile(obs, (self.N,1)).T)
		self.U.append(u)
		self.goals.append(goal)
		throttle, brake = self.Get_Carla_Throttle_Input(u[0,1])
		steer = self.Get_Carla_Steer_Input(u[1,1])
		print('action: ', u[:, 1], 'state: ', x_mean)

		return [throttle, brake, steer], self.X_pred[-1], self.X_pred_obs[-1], exitflag, False


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

	def loss(self, x, x_o, r):
		return r**2 - (x_o[:self.ny] - x[:self.ny]).T @ (x_o[:self.ny] - x[:self.ny])


	def dynamics(self, z):
		x = z[2*self.nu:2*self.nu+self.nx]
		u = z[self.nu:2*self.nu]
		x_new = casadi.SX.sym('x_new', self.nu + self.nx)
		deltau = z[:self.nu]
		x_new[:2] = deltau + u
		beta = casadi.atan(self.l_r * casadi.tan(x_new[1])/ self.L)
		x_new[2] = x[0] + self.Ts*x[3]*casadi.cos(x[2] + beta)
		x_new[3] = x[1] + self.Ts*x[3]*casadi.sin(x[2] + beta)
		x_new[4] = x[2] + self.Ts*casadi.tan(x_new[1])*casadi.cos(beta)*x[3]/self.L
		x_new[5] = x[3] + self.Ts*x_new[0]
		return x_new
		
	def objective(self, z, p):
		x = z[2*self.nu:2*self.nu+self.nx]
		deltau = z[:self.nu]
		goal = p[:self.nx]
		Q = casadi.diag(p[self.nx:2*self.nx])
		R = casadi.diag(p[2*self.nx: 2*self.nx+self.nu])
		obj = casadi.dot((Q @ (x[:self.nx] - goal)), x[:self.nx] - goal) + casadi.dot((R @ deltau), deltau)
		return obj
		
	def objectiveN(self, z, p):
		x = z[2*self.nu:2*self.nu+self.nx]
		goal = p[:self.nx]
		Qf = casadi.diag(p[2*self.nx+self.nu: 3*self.nx+self.nu])
		obj = casadi.dot((Qf @ (x[:self.nx] - goal)), x[:self.nx] - goal)
		return obj

	def inequality(self, z, p):
		x = z[2*self.nu:2*self.nu+self.nx]
		x_o = p[3*self.nx + self.nu + 1:3*self.nx + self.nu+self.ny + 1]
		r = p[3*self.nx + self.nu]
		return [self.loss(x, x_o, r)]

	def create_MPC(self):
		model = forcespro.nlp.SymbolicModel(self.N)
		model.N = self.N
		model.nvar = 2*self.nu + self.nx
		model.neq = self.nu + self.nx
		model.npar = self.nx + 2*self.nx + self.nu + self.ny + 1
		model.nh = 1
		model.ineq = self.inequality
		model.hu = [0]
		model.objective = self.objective
		model.objectiveN = self.objectiveN
		model.eq = self.dynamics
		model.E = np.concatenate([np.zeros((self.nu+self.nx, self.nu)), np.eye(self.nu+self.nx)], axis=1)
		model.ubidx = range(self.nu-1, 2*self.nu)
		model.lbidx = range(self.nu-1, 2*self.nu)
		model.xinitidx = range(self.nu, 2*self.nu+self.nx)
		codeoptions = forcespro.CodeOptions('Vanilla_MPC')
		codeoptions.maxit = 2000
		codeoptions.printlevel = 0
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
		#ax.scatter(world.waypoints_array[:,0], world.waypoints_array[:,1], color='tab:gray', marker='.', s=1)
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
		line2, = ax.plot(x, y, 'r--', label='MPC Prediction')
		goal_pl, = ax.plot(x, y, color='tab:blue')
		line_r = ax.scatter(x, y, label='Car State', marker='.', color = 'tab:blue')
		line_o = ax.scatter(x, y, label='Obstacle State', marker='.', color = 'tab:green')
		line_oo, = ax.plot(x, y, label='Obstacle Box', color = 'tab:green')
		line_rr, = ax.plot(x, y, label='Car Box', color = 'tab:blue')
		timestep = ax.text(-5.5, -5.7, '', fontsize=15)
		plt.gca().invert_yaxis()
		def func(t):
			x_traj = self.X_mean[:t,0]
			y_traj = self.X_mean[:t,1]

			timestep.set_text('t = {}'.format(t))
			line_r.set_offsets(self.X_mean[t,:2])
			line_o.set_offsets(self.obstacles[t][:2])

			line.set_data(x_traj, y_traj)
			line2.set_data(self.X_pred[t][0, :], self.X_pred[t][1, :])
			if self.with_obs:
				line_oo.set_data(self.obs_box[t][:,0], self.obs_box[t][:,1])
			line_rr.set_data(self.car_box[t][:,0], self.car_box[t][:,1])
			goal_pl.set_data(self.goals[t][:,0], self.goals[t][:,1])

		# Animate
		ani = FuncAnimation(fig=fig, func=func, frames=T)
		plt.grid()

		plt.scatter(self.X_mean[0,0], self.X_mean[0,1], color='tab:blue')
		plt.annotate('starting position', tuple(self.X_mean[0,:2]))
#		ax.axis([-140, 50, -80, 50])
		plt.legend()

		ani.save(self.path + 'vid_{}.mp4'.format(time.strftime("%m%d-%H%M%S")), writer=writer)
		plt.close('all')
		print('---------Done!-------')
