from sys import platform as sys_pf
import matplotlib
if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
import numpy as np

from main import load_input

p1 = np.array([1, 2, 3])
p2 = np.array([4, 6, 9])
p3 = np.array([12, 11, 9])

# These two vectors are in the plane
v1 = p3 - p1
v2 = p2 - p1

# the cross product is a vector normal to the plane
cp = np.cross(v1, v2)
a, b, c = cp

# This evaluates a * x3 + b * y3 + c * z3 which equals d
d = np.dot(cp, p3)

print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def normalize(v):
    mn = -2 #np.min(v)
    mx = +3 #np.max(v)
    v = (v - mn) / (mx - mn)
    return v

if True:
    param_p, points = load_input("input.txt")
    color = [cm.jet(x) for x in normalize(points[:, 2])]
    # ax.plot(*[points[:, i] for i in range(3)],
    #         color=color, linestyle=' ', marker='.', markersize=1.0)
    ax.scatter(*[points[:, i] for i in range(3)],
               c=color, s=1.0)
    plt.tight_layout()
    plt.show()
else:
    x = np.linspace(-2, 14, 5)
    y = np.linspace(-2, 14, 5)
    X, Y = np.meshgrid(x, y)

    Z = (d - a * X - b * Y) / c

    # plot the mesh. Each array is 2D, so we flatten them to 1D arrays
    ax.plot(X.flatten(),
            Y.flatten(),
            Z.flatten(), 'bo ')

    # plot the original points. We use zip to get 1D lists of x, y and z
    # coordinates.
    ax.plot(*zip(p1, p2, p3), color='r', linestyle=' ', marker='o')

    # adjust the view so we can see the point/plane alignment
    ax.view_init(0, 22)
    plt.tight_layout()
    plt.savefig('plane.png')
    plt.show()
