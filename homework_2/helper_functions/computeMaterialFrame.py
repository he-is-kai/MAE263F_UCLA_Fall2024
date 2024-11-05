import numpy as np


def computeMaterialFrame(a1, a2, theta):
    ne = len(theta)
    m1 = np.zeros((ne, 3))
    m2 = np.zeros((ne, 3))
    for c in range(ne):  # loop over every edge
        m1[c, :] = a1[c, :] * np.cos(theta[c]) + a2[c, :] * np.sin(theta[c])
        m2[c, :] = -a1[c, :] * np.sin(theta[c]) + a2[c, :] * np.cos(theta[c])
    return m1, m2
