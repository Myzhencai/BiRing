from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import numpy as np

# center and radius
center = [0, 0, 0]
radius = 100

# data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

# plot
fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
ax = fig.add_subplot( projection='3d')

# surface plot rstride 值越大，图像越粗糙
# ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')

# wire frame
# ax = fig.add_subplot(122, projection='3d')
# ax.plot_wireframe(x, y, z, rstride=10, cstride=10)

calibratex =x.reshape((10000,))
calibratey =y.reshape((10000,))
calibratez =z.reshape((10000,))
# print("x",x[0].shape)
# print("y",y[0].shape)
# print("z",z[0].shape)
ax.scatter3D(calibratex, calibratey, calibratez, color="red")


# show
plt.show()