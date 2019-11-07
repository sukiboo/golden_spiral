
''' generate binary Van der Corput sequence '''

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, linewidth=115, suppress=True, formatter={'float':'{: 0.3f}'.format})
import measure_efficiency as me
import time
start_time = time.time()


num_pts = 1000000
base = 2

pts = np.array([])
DD = np.array([])
iter = 0
for n in range(num_pts):
	n_th_number, denom = 0., 1.
	while n > 0:
		n, remainder = divmod(n, base)
		denom *= base
		n_th_number += remainder / denom
	pts = np.append(pts, n_th_number)
	DD = np.append(DD, me.compute_discrepancy(pts))
	iter += 1
	print('iteration {:d}/{:d} completed, elapsed time: {:.2f}s'\
		.format(iter, num_pts, time.time() - start_time))
print('total elapsed time: {:.2f}s'.format(time.time() - start_time))


plt.figure(figsize=(19,8))
ax = plt.axes()

x_pts = np.arange(num_pts) + 1
ax.plot(x_pts, DD * x_pts, color='turquoise', label='nD*-discrepancy')
Dy = np.log(x_pts) / (3*np.log(2)) + 1
ax.plot(x_pts, Dy, color='crimson', label='theoretical estimate')

ax.legend(loc='upper left')
plt.tight_layout()
plt.show()


import pickle
# #pickle.dump([num_pts, base, x_pts, DD, Dy], open('vars/1mil_vdc', 'wb'))
# #num_pts, base, x_pts, DD, Dy = pickle.load(open('vars/1mil_vdc', 'rb'))


plt.figure(figsize=(19,8))
ax = plt.axes()

Dy = np.log(x_pts) / (3*np.log(2)) + 1
ax.plot(x_pts, DD * x_pts, color='turquoise', label='nD*-vdc')
ax.plot(x_pts, DD2 * x_pts, color='dodgerblue', label='nD*-phi')
ax.plot(x_pts, Dy, color='crimson', linewidth=2, label='theoretical estimate')

ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

