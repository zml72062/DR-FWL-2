import numpy as np
from typing import Dict, Set

class AdjList:
    """
    Maps nodes to adjacent nodes.
    """
    def __init__(self):
        self.num_nodes = 0
        self.num_edges = 0
        self.adj_dict: Dict[int, Set[int]] = {}
    
    def has_edge(self, src: int, tgt: int) -> bool:
        return src in self.adj_dict and tgt in self.adj_dict[src]

    def add_edge(self, src: int, tgt: int, undirected: bool = True):
        self._add_directed_edge(src, tgt)
        if undirected:
            self._add_directed_edge(tgt, src)

    def _add_directed_edge(self, src: int, tgt: int):
        self.num_nodes = max(self.num_nodes, src + 1, tgt + 1)
        if self.has_edge(src, tgt):
            pass
        elif src not in self.adj_dict:
            self.adj_dict[src] = {tgt}
            self.num_edges += 1
        else:
            self.adj_dict[src].add(tgt)
            self.num_edges += 1
    
    @staticmethod
    def from_dense_adj(A: np.ndarray) -> "AdjList":
        assert len(A.shape) == 2
        return AdjList.from_sparse_adj(np.stack(np.where(A)))
    
    @staticmethod
    def from_sparse_adj(edge_index: np.ndarray) -> "AdjList":
        assert len(edge_index.shape) == 2 and edge_index.shape[0] == 2
        adj = AdjList()
        for i in range(edge_index.shape[1]):
            adj._add_directed_edge(edge_index[0, i], edge_index[1, i])
        return adj
    
    def to_sparse_adj(self) -> np.ndarray:
        edge_list = []
        for src in self.adj_dict:
            for tgt in self.adj_dict[src]:
                edge_list.append(np.array([src, tgt], dtype=np.int64))
        return np.stack(edge_list).T
