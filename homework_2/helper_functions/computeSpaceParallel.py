import numpy as np
from .computeTangent import computeTangent
from .parallel_transport import parallel_transport


def computeSpaceParallel(d1_first, q):
    ne = int((len(q) + 1) / 4 - 1)
    tangent = computeTangent(q)

    d1 = np.zeros((ne, 3))
    d2 = np.zeros((ne, 3))

    # First edge
    d1[0, :] = d1_first  # Given
    t0 = tangent[0, :]  # Tangent on first edge
    d2[0, :] = np.cross(t0, d1_first)

    # Parallel transport from previous edge to the next
    for c in range(1, ne):
        t = tangent[c, :]
        d1_first = parallel_transport(d1_first, t0, t)
        # d1_first should be perpendicular to t
        d1_first = d1_first - np.dot(d1_first, t) * t
        d1_first = d1_first / np.linalg.norm(d1_first)

        # Store d1 and d2 vectors for c-th edge
        d1[c, :] = d1_first
        d2[c, :] = np.cross(
            t, d1_first
        )  # I made a mistake in class and wrote cross(t0, d1_first)

        t0 = t.copy()  # New tangent now becomes old tangent
        # I made a mistake in class and forgot to write "t.copy()" and just wrote "t0=t"

    return d1, d2
