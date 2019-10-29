
''' plot Dg_n(gamma) as a function of n '''

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, linewidth=115, suppress=True, formatter={'float':'{: 0.3f}'.format})
import measure_efficiency as me

gamma = (1 + 5**.5) / 2
K_max = 1

num_pts = 1000
step_pts = 100

x_pts = (1 + np.arange(num_pts)) * step_pts
Dt = ((K_max + 1)**4 / 2 + K_max + 1) / x_pts

Dg = np.zeros(num_pts)
Dm = np.zeros(num_pts)
DD = np.zeros(num_pts)

for n in range(num_pts):
	pts = np.arange(x_pts[n]) * gamma % 1
	DD[n] = me.compute_discrepancy(pts)
	pts = np.concatenate((np.sort(pts), [1]))
	delta = pts[1:] - pts[:-1]
	Dg[n] = np.max(delta)
	Dm[n] = np.min(delta, initial=1, where=(delta > 1e-08))

plt.figure(figsize=(19,8))
ax = plt.axes()
ax.set_xlim(0, num_pts*step_pts)
# #ax.set_xticks(np.linspace(0, num_pts, 11))
# #ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
# #ax.set_ylim(0, 1)
# #ax.set_yticks(np.linspace(0, 1, 11))
ax.set_yscale('log')

ax.plot(x_pts, DD, color='turquoise', label='D*-discrepancy')
ax.plot(x_pts, Dg, color='dodgerblue', label='maximal gap')
ax.plot(x_pts, Dm, color='darkorange', label='minimal gap')
ax.plot(x_pts, Dt, color='crimson', label='theoretical estimate')

ax.legend(loc='upper right')

plt.tight_layout()
plt.show()



plt.figure(figsize=(19,8))
ax = plt.axes()
ax.plot(x_pts, DD * x_pts, color='turquoise', label='nD*-discrepancy')
Dy = np.log(x_pts) / np.log(2) / 4 + 1
ax.plot(x_pts, Dy, color='crimson', label='theoretical estimate')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()


'''
pts = np.tril(np.tile(np.arange(num_pts), (num_pts,1))) * gamma % 1
pts = np.concatenate((np.sort(pts), np.ones((num_pts,1))), axis=1)

delta = pts[:,1:] - pts[:,:-1]
Dg = np.max(delta, axis=1)
Dm = np.min(delta, axis=1, initial=1, where=(delta > 1e-08))

plt.figure(figsize=(19,8))
ax = plt.axes()

# #ax.set_xlim(0, num_pts)
# #ax.set_xticks(np.linspace(0, num_pts, 11))
# #ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

# #ax.set_ylim(0, 1)
# #ax.set_yticks(np.linspace(0, 1, 11))
ax.set_yscale('log')

ax.plot(np.arange(num_pts), Dg, color='dodgerblue')
ax.plot(np.arange(num_pts), Dm, color='darkorange')
ax.plot(np.arange(num_pts), 10/np.arange(num_pts), color='crimson')
'''
