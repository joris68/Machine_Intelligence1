from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import numpy as np

expdat = np.genfromtxt(fname='EX0/applesOranges.csv', delimiter=',', 
                       dtype=float, skip_header=1)
x = expdat[:, :2]
y = expdat[:, -1]

def binedges2center(edges_x, edges_y):
    xcenters = (edges_x[:-1] + edges_x[1:]) / 2.
    ycenters = (edges_y[:-1] + edges_y[1:]) / 2.
    return xcenters, ycenters

hist_0, bin_edges_x_0, bin_edges_y_0 = np.histogram2d(x[y==0, 0], x[y==0, 1],
                                                         bins=10, density=True)
hist_1, bin_edges_x_1, bin_edges_y_1 = np.histogram2d(x[y==1, 0], x[y==1, 1],
                                                         bins=10, density=True)

bins_x_0, bins_y_1 = binedges2center(bin_edges_x_0, bin_edges_y_0)
bins_x_1, bins_y_1 = binedges2center(bin_edges_x_1, bin_edges_y_1)

fig, ax = plt.subplots(1, figsize=[7, 7])
# min and max values in all directions
vmax = max(np.abs(hist_0).max(), np.abs(hist_1).max())
xmin = min(bin_edges_x_0.min(), bin_edges_x_1.min())
xmax = max(bin_edges_x_0.max(), bin_edges_x_1.max())
ymin = min(bin_edges_y_0.min(), bin_edges_y_1.min())
ymax = max(bin_edges_y_0.max(), bin_edges_y_1.max())

# interpolate for smoother density-like plot
xi2 = np.linspace(xmin, xmax, 50)
yi2 = np.linspace(ymin, ymax, 50)
f = interp2d(bin_edges_x_0[:-1], bin_edges_y_0[:-1], hist_0, kind='cubic')
hist_0_smooth = f(xi2, yi2)
f = interp2d(bin_edges_x_1[:-1], bin_edges_y_1[:-1], hist_1, kind='cubic')
hist_1_smooth = f(xi2, yi2)

step = 0.0001
levels = np.arange(0.0, vmax, step) + step

cmap = plt.cm.Reds
# proxy = [plt.Rectangle((0,0),1,1, fc=cmap(1))]
ax.contourf(hist_0_smooth, cmap=cmap, alpha=0.5)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$', rotation='horizontal')
cmap = plt.cm.Blues
# proxy.append(plt.Rectangle((0,0),1,1, fc=cmap(1)))
ct = ax.contourf(hist_1_smooth, cmap=cmap, alpha=0.5)
plt.grid()

import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='apples')
blue_patch = mpatches.Patch(color='blue', label='oranges')
plt.legend(handles=[red_patch, blue_patch])

plt.show()
