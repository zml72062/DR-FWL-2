from furer import get_furer_graph_pair
import numpy as np
from networkx import Graph, is_isomorphic

g, gx, h, hx = get_furer_graph_pair(
    np.array(
        [[0, 1, 1, 2, 2, 3, 3, 0, 1, 3, 0, 2],
         [1, 0, 2, 1, 3, 2, 0, 3, 3, 1, 2, 0]], dtype=np.int64
    ) # build furer graph from 4-clique
)

rook = []
for i in range(16):
    for j in range(i + 1, 16):
        if i // 4 == j // 4 or i % 4 == j % 4:
            rook.append(np.array([i, j], dtype=np.int64))
            rook.append(np.array([j, i], dtype=np.int64))
rook = np.stack(rook).T

def mod4is1(i, j):
    return i % 4 == (j + 1) % 4

shrikhande = []
for i in range(16):
    for j in range(i + 1, 16):
        if i // 4 == j // 4 and (mod4is1(i % 4, j % 4) or 
                                 mod4is1(j % 4, i % 4)) or\
           i % 4 == j % 4 and (mod4is1(i // 4, j // 4) or
                               mod4is1(j // 4, i // 4)) or\
           mod4is1(i % 4, j % 4) and mod4is1(i // 4, j // 4) or\
           mod4is1(j % 4, i % 4) and mod4is1(j // 4, i // 4):
            shrikhande.append(np.array([i, j], dtype=np.int64))
            shrikhande.append(np.array([j, i], dtype=np.int64))
shrikhande = np.stack(shrikhande).T

def test_iso(x, y):
    g = Graph()
    g.add_edges_from(x.T)
    h = Graph()
    h.add_edges_from(y.T)
    return is_isomorphic(g, h)


print(test_iso(g, rook)) # True
print(test_iso(h, shrikhande)) # True
print(test_iso(g, h))