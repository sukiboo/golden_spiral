

import measure_efficiency as me
import plot_points
import construct
import compare
import time
import numpy as np
np.set_printoptions(precision=3, linewidth=115, suppress=True, formatter={'float':'{: 0.3f}'.format})


''' input setting '''
dim = 2
obj = 's'
num_points = 1000


''' initialize the variables '''
# check the problem setting
obj = ('sphere' if obj=='s' else 'ball')
print('\nselecting {:d} points from {:d}-{:s}'.format(num_points, dim, obj))

# find the corresponding generalized golden ratio
d_phi = construct.generalized_golden_ratio(dim)
seed = np.array([d_phi**n for n in range(1,dim)])

# define golden ratio
phi = (np.sqrt(5) + 1) / 2
print('phi = {:f}\n'.format(phi))


''' generate low-discrepancy sequences with multiple seeds'''
seq = construct.ggs_sequences(seed, num_points)

''' measure the sequence efficiency '''
me.sequence_correlation(seq)
me.sequence_discrepancy(seq)

''' construct the points '''
points = construct.ggs_points(obj, seq)

''' analyze and plot the points '''
st_ggs = me.display_stats(dim, obj, points)
plot_points.plot_points(dim, obj, points, animate=False, save=False)

''' compare to random points '''
st_rnd = compare.random_points(dim, obj, num_points, animate=False, save=False)

