
''' interactively plot Dg_n(gamma) as a function of gamma '''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
np.set_printoptions(precision=3, linewidth=115, suppress=True, formatter={'float':'{: 0.3f}'.format})


num_points = 2
num_seed = 10000

# compute largest gap discrepancy
def compute_D0(num_points, num_seed):
    seed = np.linspace(0, 1, num_seed)
    pts = np.arange(num_points) * seed.reshape((-1,1)) % 1
    pts = np.concatenate((np.sort(pts), np.ones((seed.shape[0],1))), axis=1)
    D0 = np.max(pts[:,1:] - pts[:,:-1], axis=1)
    return D0

# set up figure and configure axes
fig = plt.figure(figsize=(10,11))
plt.subplots_adjust(left=.07, right=.97, bottom=.15, top=.97)
def plot_lattice(num_points, num_seed, color='dodgerblue'):
    ax = plt.axes()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(.05))
    ax.set_xlabel('value of gamma')
    ax.set_ylabel('largest-gap discrepancy')
    # plot the points
    D0 = compute_D0(num_points, num_seed)
    plot, = ax.plot(np.linspace(0, 1, num_seed), D0, color=color)
    return plot,

plot, = plot_lattice(num_points, num_seed)
# buttons that change the number of points
num_minus_axes = plt.axes([.79, .03, .08, .05])
num_minus_button = Button(num_minus_axes, '-1')
num_plus_axes = plt.axes([.89, .03, .08, .05])
num_plus_button = Button(num_plus_axes, '+1')
# number of points
num_points_axes = plt.axes([.69, .03, .08, .05])
points_button = TextBox(num_points_axes, 'number of points:', initial=str(num_points), label_pad=.1)

# update the plot on a slider update
def slider_update(_):
    D0 = compute_D0(int(points_button.text), num_seed)
    plot.set_ydata(D0)
    fig.canvas.draw_idle()
def points_minus(_):
    points_button.set_val(int(points_button.text) - 1)
    points_button.stop_typing()
def points_plus(_):
    points_button.set_val(int(points_button.text) + 1)
    points_button.stop_typing()

# update the plot
points_button.on_submit(slider_update)
num_minus_button.on_clicked(points_minus)
num_plus_button.on_clicked(points_plus)

# show the plot
plt.show()
