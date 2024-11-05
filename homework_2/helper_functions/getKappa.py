import numpy as np
from .computekappa import computekappa


def getKappa(q0, m1, m2):
    ne = m1.shape[0]  # Shape of m1 is (ne,3)
    nv = ne + 1

    kappa = np.zeros((nv, 2))

    for c in np.arange(1, ne):
        node0 = q0[4 * c - 4 : 4 * c - 1]
        node1 = q0[4 * c + 0 : 4 * c + 3]
        node2 = q0[4 * c + 4 : 4 * c + 7]

        m1e = m1[c - 1, 0:3].flatten()  # Material frame of previous edge
        m2e = m2[c - 1, 0:3].flatten()  # NOT SURE if flattening is needed or not
        m1f = m1[c, 0:3].flatten()  # Material frame of current edge
        m2f = m2[c, 0:3].flatten()

        kappa_local = computekappa(node0, node1, node2, m1e, m2e, m1f, m2f)

        # Store the values
        kappa[c, 0] = kappa_local[0]
        kappa[c, 1] = kappa_local[1]

    return kappa
