

import numpy as np
import scipy
import sympy as sym



''' find the d-golden ratio '''
def generalized_golden_ratio(d, max_iter=1000):

	# solve the equation x**d = x + 1
	x, dx, i = 2, 2, 0

	try:
		# solve via newton method
		while (np.abs(dx) > 1e-16) and (i < max_iter):
			dx = (x**d - x - 1) / (d*x**(d-1) - 1)
			x -= dx
			i += 1

	except:
		# solve via recursion x = (1+x)**(1/d)
		while (np.abs(dx) > 1e-16) and (i < max_iter):
			dx = x - (1 + x)**(1/d)
			x -= dx
			i += 1

	# display the solution
	print('generalized golden ratio d_phi = {:.15f}'.format(x))

	return x


''' generate low-discrepancy sequences '''
def ggs_sequences(d_phi, num_points):

	# generate 'independent' uniformly distributed sequences
	ind = np.arange(num_points)
	seq = np.zeros((d_phi.size+1, num_points))
	for n in range(d_phi.size):
		seq[n] = ind * d_phi[n] % 1
	seq[-1] = (ind + .5) / num_points

	return seq


''' construct the points on the object (obj) based on the provided sequences (seq) '''
def ggs_points(obj, seq):

	# initialize the variables
	dim, num_points = seq.shape

	# d-sphere
	if obj == 'sphere':
		# angles
		# #I = compute_I_n(dim)
		# #pts = invert_distribution(I, seq, dim)
		pts = invert_distribution(dim, seq)
		# convert to cartesian coordinates
		pts = spherical_to_cartesian(pts)

	# d-ball
	elif obj == 'ball':
		# angles
		# #I = compute_I_n(dim-1)
		# #pts = invert_distribution(I, seq, dim-1)
		pts = invert_distribution(dim-1, seq)
		# radius
		pts[-1] = seq[-1]**(1/dim)
		# convert to cartesian coordinates
		pts = spherical_to_cartesian(pts[:-1], pts[-1])

	return pts.T


''' construct the points from the given values of angles (psi) and radii (rho) '''
def spherical_to_cartesian(psi, rho=1):

	# extract dimensionality and number of points
	psi_dim, num_points = psi.shape

	# compute cartesian coordinates
	points = np.ones((psi_dim+1, num_points))
	for n in range(psi_dim)[::-1]:
		points[:n+1] *= np.sin(psi[n])
		points[n+1] *= np.cos(psi[n])
	points *= rho

	return points


''' find the distribution of (pts) so that I[n](pts[n]) = I[n](pi) * seq[n] '''
def invert_distribution(dim, seq):

	# extract dimensionality and number of points
	dim, num_points = seq.shape
	pts = np.zeros(seq.shape)

	# explicitly invert the first two sequences
	pts[0] = seq[0] * 2*np.pi
	pts[1] = np.arccos(1 - 2*seq[1])
	print('invert variable {:3d}/{:d}, accuracy: {:.2e}'\
		.format(1, dim, np.max(np.abs(pts[0]/(2*np.pi) - seq[0]))))
	print('invert variable {:3d}/{:d}, accuracy: {:.2e}'\
		.format(2, dim, np.max(np.abs((1 - np.cos(pts[1]))/2 - seq[1]))))

	# initial guess for the next sequence
	s_init = np.linspace(0, 1, num_points, endpoint=False)
	p_init = np.arccos(1 - 2*s_init)

	# implicily invert the later ones
	for n in range(2,dim):

		# define the function and its derivative
		x = sym.symbols('x')
		I = sym.integrate(sym.sin(x)**n, (x, 0, x)) / sym.integrate(sym.sin(x)**n, (x, 0, np.pi))
		f = sym.lambdify(x, I)
		df = sym.lambdify(x, sym.sin(x)**n)

		# invert the function
		pts[n] = invert_function(f, df, n, seq[n], p_init)
		p_init = invert_function(f, df, n, s_init, p_init)
		print('invert variable {:3d}/{:d}, accuracy: {:.2e}'\
			.format(n+1, dim, np.max(np.abs(f(pts[n]) - seq[n]))))

	return pts


''' find such x that f(x) = y '''
def invert_function(f, df, n, y, x0):

	# direct and inverse rearrangings
	num_points = y.size
	ind_dir = y.argsort()
	ind_inv = np.arange(num_points)[ind_dir].argsort()
	# sorted and scale the sequence
	s = y[ind_dir]

	# split the sequence into two halves
	eps = 1e-04
	s0, s2 = np.split(s, [num_points//2])

	# approximate sin(x)**n by x**n on the first half
	p0 = ((n+1) * s0)**(1/(n+1))
	n0 = max(np.searchsorted(s0-f(p0), eps), int(.001*num_points))

	# approximate sin(x)**n by (pi-x)**n on the second half
	p2 = np.pi - ((n+1) * (1 - s2))**(1/(n+1))
	n2 = max(np.searchsorted((f(p2)-s2)[::-1], eps), int(.001*num_points))

	# re-split the sequence
	s0, s1, s2 = np.split(s, [n0,num_points-n2])
	p0, p2 = p0[:n0], p2[-n2:]

	# invert distribution by newton method on the middle interval
	p1 = scipy.optimize.newton(\
			lambda x: f(x) - s1, \
			np.sort(x0)[n0 : n0+s1.size],
			# #np.sort(pts[n-1])[num_x : num_x+s1.size],
			fprime = lambda x: df(x),
			maxiter = 1000)

	# #if s0.size > 0:
		# #print('p0: {:.2e}, {:d}'.format(np.max(np.abs(f(p0) - s0)), p0.size))
	# #print('p1: {:.2e}, {:d}'.format(np.max(np.abs(f(p1) - s1)), p1.size))
	# #if s2.size > 0:
		# #print('p2: {:.2e}, {:d}'.format(np.max(np.abs(f(p2) - s2)), p2.size))

	# merge and rearrange the points
	x = np.concatenate([p0,p1,p2])[ind_inv]

	return x


''' find the upper d-golden ratio '''
def generalized_upper_golden_ratio(d, max_iter=1000):

	# solve the equation x*(x-1)**(d-1) = 1
	x, dx, i = 2, 2, 0

	try:
		# solve via newton method
		while (np.abs(dx) > 1e-16) and (i < max_iter):
			dx = (x*(x-1)**(d-1) - 1) / ((d*x - 1) * (x-1)**(d-2))
			x -= dx
			i += 1

	except:
		# solve via recursion x = (x-1)**(1-d)
		while (np.abs(dx) > 1e-16) and (i < max_iter):
			dx = x - (x-1)**(1-d)
			x -= dx
			i += 1

	# display the solution
	print('generalized golden ratio d_phi = {:.15f}'.format(x))

	return x


def generate_lattice(step, alpha, num_points):
	seq = np.zeros((2,num_points))
	ind = np.arange(num_points) * step
	seq[0] = ind * np.cos(alpha * np.pi/180) % 1
	seq[1] = ind * np.sin(alpha * np.pi/180) % 1
	return seq

def interactive_lattice(seq, alpha, step):

	num_points = seq.shape[1]

	import matplotlib.pyplot as plt
	from matplotlib.widgets import Slider, Button, TextBox
	from matplotlib import animation

	# set up figure and configure axes
	fig = plt.figure(figsize=(10,11))
	plt.subplots_adjust(left=.05, right=.95, bottom=.15, top=.96)
	def plot_lattice(pts0, pts1, color='dodgerblue', save=False):
		ax = plt.axes()
		ax.set_xlim(-.01, 1.01)
		ax.set_ylim(-.01, 1.01)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot the points
		plot = ax.scatter(pts0, pts1, color=color)
		xtick = ax.scatter(pts0, np.zeros(pts0.shape)-.005, color='firebrick', marker='|')
		ytick = ax.scatter(np.zeros(pts1.shape)-.005, pts1, color='firebrick', marker='_')
		return plot, xtick, ytick

	plot, xtick, ytick = plot_lattice(seq[0], seq[1])
	alpha_axes = plt.axes([.075, .1, .85, .025])
	alpha_slider = Slider(alpha_axes, 'angle', 0, 45, valinit=alpha)
	step_axes = plt.axes([.075, .06, .85, .025])
	step_slider = Slider(step_axes, 'step', 0, np.sqrt(2), valinit=step)
	# buttons that change the number of points
	num_minus_axes = plt.axes([.78, .01, .065, .04])
	num_minus_button = Button(num_minus_axes, '-1')
	num_plus_axes = plt.axes([.86, .01, .065, .04])
	num_plus_button = Button(num_plus_axes, '+1')
	# number of points
	num_points_axes = plt.axes([.7, .01, .065, .04])
	points_button = TextBox(num_points_axes, 'number of points:', initial=str(num_points), label_pad=.1)

	# update the plot on a slider update
	def slider_update(_):
		seq = generate_lattice(step_slider.val, alpha_slider.val, int(points_button.text))
		plot.set_offsets(seq.T)
		xtick.set_offsets(np.array([seq[0], np.zeros(seq[0].shape)-.005]).T)
		ytick.set_offsets(np.array([np.zeros(seq[1].shape)-.005, seq[1]]).T)
		fig.canvas.draw_idle()
	def points_minus(_):
		points_button.set_val(int(points_button.text) - 1)
		points_button.stop_typing()
	def points_plus(_):
		points_button.set_val(int(points_button.text) + 1)
		points_button.stop_typing()

	# update the plot
	alpha_slider.on_changed(slider_update)
	step_slider.on_changed(slider_update)
	points_button.on_submit(slider_update)
	num_minus_button.on_clicked(points_minus)
	num_plus_button.on_clicked(points_plus)

	# show the plot
	plt.show()

	# update and return the sequence
	seq = generate_lattice(step_slider.val, alpha_slider.val, int(points_button.text))
	return seq

def seq_ratio(N, r):
	seq = np.zeros(N)
	for n in range(N-1):
		seqs = np.concatenate([np.sort(seq[:n+1]), (1,)])
		seqs_d = seqs[1:] - seqs[:-1]
		k = np.argmax(seqs_d)
		seq[n+1] = (1-r)*seqs[k] + r*seqs[k+1]
	return seq
