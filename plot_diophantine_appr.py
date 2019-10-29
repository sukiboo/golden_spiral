import numpy as np
import matplotlib.pyplot as plt


num_pts = 1000
gamma = (1 + np.sqrt(5)) / 2

pts = np.arange(1, num_pts+1)

ratio = (pts*gamma % 1) * pts / np.log(pts+1)
# #ratio1 = np.abs(gamma*pts - np.ceil(gamma*pts)) * pts / np.log(pts+1)
# #ratio2 = np.abs(gamma*pts - np.floor(gamma*pts)) * pts / np.log(pts+1)
# #ratio = np.minimum(ratio1, ratio2)


plt.figure(figsize=(19,6))
plt.plot(pts, ratio)
# #plt.plot(pts, ratio1)
# #plt.plot(pts, ratio2)
plt.show()

