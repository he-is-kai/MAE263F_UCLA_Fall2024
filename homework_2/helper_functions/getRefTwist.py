import numpy as np
from .computeReferenceTwist import computeReferenceTwist


def getRefTwist(a1, tangent, refTwist):
    ne = a1.shape[0]  # Shape of a1 is (ne,3)
    for c in np.arange(1, ne):
        u0 = a1[c - 1, 0:3]  # reference frame vector of previous edge
        u1 = a1[c, 0:3]  # reference frame vector of current edge
        t0 = tangent[c - 1, 0:3]  # tangent of previous edge
        t1 = tangent[c, 0:3]  # tangent of current edge
        refTwist[c] = computeReferenceTwist(u0, u1, t0, t1, refTwist[c])
    return refTwist
