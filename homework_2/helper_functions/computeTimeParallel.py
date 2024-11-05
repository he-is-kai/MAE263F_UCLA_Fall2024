import numpy as np
from .computeTangent import computeTangent
from .parallel_transport import parallel_transport


def computeTimeParallel(a1_old, q0, q):
    # a1_old is (ne,3) ndarray representing old reference frame
    # q0 is the old DOF vector from where reference frame should be transported
    # q is the new DOF vector where reference frame should be transported to
    ne = int((len(q) + 1) / 4 - 1)
    tangent0 = computeTangent(q0)  # Old tangents
    tangent = computeTangent(q)  # New tangents

    a1 = np.zeros((ne, 3))
    a2 = np.zeros((ne, 3))
    for c in range(ne):
        t0 = tangent0[c, :]
        t = tangent[c, :]
        a1_tmp = parallel_transport(a1_old[c, :], t0, t)
        a1[c, :] = a1_tmp - np.dot(a1_tmp, t) * t
        a1[c, :] = a1[c, :] / np.linalg.norm(a1[c, :])
        a2[c, :] = np.cross(t, a1[c, :])

    return a1, a2
