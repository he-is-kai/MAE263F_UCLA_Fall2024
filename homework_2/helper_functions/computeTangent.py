import numpy as np


def computeTangent(q):
    ne = int((len(q) + 1) / 4 - 1)
    tangent = np.zeros((ne, 3))
    for c in range(ne):
        dx = q[4 * c + 4 : 4 * c + 7] - q[4 * c : 4 * c + 3]  # edge vector
        tangent[c, :] = dx / np.linalg.norm(dx)  # make it unit
    return tangent
