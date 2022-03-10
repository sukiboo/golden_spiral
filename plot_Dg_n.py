
''' plot Dg_n(gamma) as a function of n '''

import numpy as np
import matplotlib.pyplot as plt
import measure_efficiency as me
import time
np.set_printoptions(precision=3, linewidth=115, suppress=True, formatter={'float':'{: 0.3f}'.format})


# parameters
phi = (1 + np.sqrt(5)) / 2
gamma = phi
a_max = 1

num_pts = 10000
step_pts = 1

# generate points
x_pts = (1 + np.arange(num_pts)) * step_pts
K_max = 2*phi + .5 if gamma == phi else (a_max + 1)/2 * (a_max**2 + 2*a_max + 2)
Dt = K_max / x_pts

# compute discrepancy
start_time = time.time()
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
    if (n+1) % 1000 == 0:
        print('iteration {:d}/{:d} completed, elapsed time: {:.2f}s'\
            .format(n+1, num_pts, time.time() - start_time))
print('total elapsed time: {:.2f}s'.format(time.time() - start_time))

# plot various discrepancies
plt.figure(figsize=(19,8))
ax = plt.axes()
ax.set_xlim(0, num_pts*step_pts)
ax.set_yscale('log')
ax.plot(x_pts, DD, color='turquoise', label='D*-discrepancy')
ax.plot(x_pts, Dg, color='dodgerblue', label='maximal gap')
ax.plot(x_pts, Dm, color='darkorange', label='minimal gap')
ax.plot(x_pts, Dt, color='crimson', label='theoretical estimate')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

# plot discrepancy
plt.figure(figsize=(19,8))
ax = plt.axes()
ax.plot(x_pts, DD * x_pts, color='turquoise', label='nD*-discrepancy')
Dy = np.log(x_pts) / (3*np.log(2)) + 1
ax.plot(x_pts, Dy, color='crimson', label='theoretical estimate')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
