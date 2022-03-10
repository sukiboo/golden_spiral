

import measure_efficiency as me
import plot_points
import numpy as np
np.set_printoptions(precision=3, linewidth=115, suppress=True, formatter={'float':'{: 0.3f}'.format})


def random_points(dim, obj, num_points, animate=False, save=False):

    d = (dim if obj=='ball' else dim+1)
    # generate num_points random points
    s_rnd = np.random.randn(num_points, d)
    s_rnd /= np.linalg.norm(s_rnd, axis=1).reshape((-1,1))
    if obj=='ball':
        s_rnd *= (np.random.uniform(0,1,(num_points,1)))**(1/d)

    ''' analyze and plot the points '''
    # display various measurements
    st = me.display_stats(dim, obj, s_rnd)
    # plot the constructed points
    plot_points.plot_points(dim, obj, s_rnd, cmap='autumn', color='orchid', animate=animate, save=save)

    return st
