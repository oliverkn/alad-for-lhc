import numpy as np


def generate_sphere_data(n, error=0):
    sphere = np.random.uniform(-1, 1, size=(n, 3))
    mag = np.linalg.norm(sphere, axis=1, ord=2)

    mag = mag * np.random.normal(1, scale=error, size=(n))

    sphere[:, 0] = sphere[:, 0] / mag
    sphere[:, 1] = sphere[:, 1] / mag
    sphere[:, 2] = sphere[:, 2] / mag
    hemisphere = np.where(sphere[:, 2] > 0, 1, 0)
    hemisphere = hemisphere.reshape((-1, 1))
    return np.concatenate([sphere, hemisphere], axis=1)

    # return sphere

def generate_ellipse_data(n, error=0):
    sphere = np.random.uniform(-1, 1, size=(n, 3))
    mag = np.linalg.norm(sphere, axis=1, ord=2)

    mag = mag * np.random.normal(1, scale=error, size=(n))

    sphere[:, 0] = sphere[:, 0] / mag
    sphere[:, 1] = sphere[:, 1] / mag / 2
    sphere[:, 2] = sphere[:, 2] / mag * 2
    hemisphere = np.where(sphere[:, 2] > 0, 1, 0)
    hemisphere = hemisphere.reshape((-1, 1))
    return np.concatenate([sphere, hemisphere], axis=1)

    # return sphere


def generate_mix(n, error=0):
    x_s = generate_sphere_data(n // 2, error=error)
    x_e = generate_ellipse_data(n // 2, error=error)

    x = np.concatenate([x_s, x_e])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

    return x, y


# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = pyplot.figure()
# ax = Axes3D(fig)
# data = generate_sphere_data(1000, error=0.1)
# ax.scatter(data[:, 0], data[:, 1], data[:, 2])
# data = generate_ellipse_data(1000, error=0.1)
# ax.scatter(data[:, 0], data[:, 1], data[:, 2])
#
# pyplot.show()
