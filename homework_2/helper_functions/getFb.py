import numpy as np
from gradEb_hessEb import gradEb_hessEb

def getFb(q, m1, m2, kappaBar, EI, voronoiRefLen):
  ndof = len(q)
  nv = int((ndof + 1) / 4 ) # Number of vertices
  ne = nv - 1 # Number of edges

  Fb = np.zeros(ndof)
  Jb = np.zeros((ndof,ndof))

  for c in range(1,ne): # Loop over all the internal nodes
    node0 = q[4 * c -4 : 4 * c - 1] # (c-1) th node
    node1 = q[4 * c : 4 * c + 3] # c-th node
    node2 = q[4 * c + 4: 4 * c + 7] # (c+1)-th node
    l_k = voronoiRefLen[c]
    kappaBar_c = kappaBar[c, :] # I made a mistake in class. I wrote kappaBar[c] instead of kappaBar[c,:]

    m1e = m1[c-1, 0:3]
    m2e = m2[c-1, 0:3]
    m1f = m1[c, 0:3]
    m2f = m2[c, 0:3]

    ind = np.arange(4*c - 4, 4 * c + 7) # 11 elements (3 nodes, 2 edges/theta angles)

    dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, kappaBar_c, l_k, EI)
    # I made a mistake in class: I got the order of the inputs wrong in gradEb_hessEb. I swapped the order of l_k and EI

    Fb[ind] -= dF
    Jb[np.ix_(ind, ind)] -= dJ

  return Fb, Jb