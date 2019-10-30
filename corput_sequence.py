
''' generate binary Van der Corput sequence '''

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, linewidth=115, suppress=True, formatter={'float':'{: 0.3f}'.format})
import measure_efficiency as me


num_pts = 1000
base = 2

pts = np.array([])
DD = np.array([])
for n in range(num_pts):
	n_th_number, denom = 0., 1.
	while n > 0:
		n, remainder = divmod(n, base)
		denom *= base
		n_th_number += remainder / denom
	pts = np.append(pts, n_th_number)
	DD = np.append(DD, me.compute_discrepancy(pts))


plt.figure(figsize=(19,8))
ax = plt.axes()

x_pts = np.arange(num_pts) + 1
ax.plot(x_pts, DD * x_pts, color='turquoise', label='nD*-discrepancy')
Dy = np.log(x_pts) / (3*np.log(2)) + 1
ax.plot(x_pts, Dy, color='crimson', label='theoretical estimate')

ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

