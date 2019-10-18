

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
num_points = 100


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

''' split gs sequence into two '''
# #seq2 = construct.ggs_sequences(seed, 2*num_points)
# #seq = np.zeros((dim,num_points))
# #seq[0] = seq2[0,0::2]
# #seq[1] = seq2[0,1::2]
# #seq[0] = seq2[0,:num_points]
# #seq[1] = seq2[0,num_points:]

''' process gs sequence twice '''
# #ss = (np.arange(num_points) * phi) % 1
# #seq[0] = (np.arcsin(1 - 2*ss) + np.pi/2) / np.pi
# #seq[1] = np.arccos(1 - 2*ss) / np.pi
# #seq[0] = (np.sin(2*np.pi * np.arange(num_points) * phi)) % 1
# #seq[1] = (np.cos(2*np.pi * np.arange(num_points) * phi)) % 1
# #seq[0] = (np.arange(num_points) * phi) % 1
# #seq[1] = (np.arange(num_points)**2 * phi) % 1

''' subinterval splitting '''
# #seq = np.zeros((2,num_points))
# #seq[0] = construct.seq_ratio(num_points, phi-1)
# #seq[1] = construct.seq_ratio(num_points, .5)
# #seq = construct.ggs_sequences(seed, num_points)
# #seq[0] = construct.seq_ratio(num_points, phi-1)

''' interactive lattice '''
# #alpha = 22# * 180/np.pi
# #step = .5
# #seq = construct.generate_lattice(step, alpha, num_points)
# #seq = construct.interactive_lattice(seq, alpha, step)


# #plot_points.plot_sequences(seq, animate=True)
plot_points.plot_lattice(seq[0], seq[1])
''' measure the sequence efficiency '''
# measure correlation and discrepancy
# #me.sequence_correlation(seq)
me.sequence_discrepancy(seq)

''' construct the points '''
points = construct.ggs_points(obj, seq)

# #''' analyze and plot the points '''
# ## display various measurements
# #st_ggs = me.display_stats(dim, obj, points)
# plot the constructed points
plot_points.plot_points(dim, obj, points, animate=False, save=False)


# #''' compare to random points '''
# #time.sleep(1)
# #st_rnd = compare.random_points(dim, obj, num_points, animate=False, save=False)


