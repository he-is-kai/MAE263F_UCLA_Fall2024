import numpy as np
from gradEt_hessEt import gradEt_hessEt_panetta as gradEt_hessEt


def getFt(q, refTwist, twistBar, GJ, voronoiRefLen):
    ndof = len(q)
    nv = int((ndof + 1) / 4)  # Number of vertices
    ne = nv - 1  # Number of edges

    Ft = np.zeros(ndof)
    Jt = np.zeros((ndof, ndof))

    for c in range(1, ne):  # Loop over all the internal nodes
        node0 = q[4 * c - 4 : 4 * c - 1]  # (c-1) th node
        node1 = q[4 * c : 4 * c + 3]  # c-th node
        node2 = q[4 * c + 4 : 4 * c + 7]  # (c+1)-th node

        theta_e = q[4 * c - 1]
        theta_f = q[4 * c + 3]

        l_k = voronoiRefLen[c]
        refTwist_c = refTwist[c]
        twistBar_c = twistBar[c]

        ind = np.arange(
            4 * c - 4, 4 * c + 7
        )  # 11 elements (3 nodes, 2 edges/theta angles)

        dF, dJ = gradEt_hessEt(
            node0, node1, node2, theta_e, theta_f, refTwist_c, twistBar_c, l_k, GJ
        )

        Ft[ind] -= dF
        Jt[np.ix_(ind, ind)] -= dJ

    return Ft, Jt
