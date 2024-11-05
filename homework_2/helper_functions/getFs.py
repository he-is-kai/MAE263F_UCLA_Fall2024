import numpy as np
from .gradEs_hessEs import gradEs_hessEs


def getFs(q, EA, refLen):
    ndof = len(q)
    nv = int((ndof + 1) / 4)  # Number of vertices
    ne = nv - 1  # Number of edges

    Fs = np.zeros(ndof)
    Js = np.zeros((ndof, ndof))

    for c in range(ne):
        node0 = q[4 * c : 4 * c + 3]
        node1 = q[4 * c + 4 : 4 * c + 7]
        ind = np.array([4 * c, 4 * c + 1, 4 * c + 2, 4 * c + 4, 4 * c + 5, 4 * c + 6])

        l_k = refLen[c]

        dF, dJ = gradEs_hessEs(node0, node1, l_k, EA)

        Fs[ind] -= dF
        Js[np.ix_(ind, ind)] -= dJ

    return Fs, Js
