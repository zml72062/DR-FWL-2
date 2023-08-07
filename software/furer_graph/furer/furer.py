import numpy as np
from typing import List, Set, Dict, Tuple
from .adj import AdjList
from .utils import *

class FurerNode:
    def __init__(self, raw_node: int,
                 node_set: Set[int]):
        self.raw_node = raw_node
        self.node_set = node_set
    
    def __repr__(self) -> str:
        return f"FurerNode({self.raw_node}, {self.node_set})"
    
    def is_connected(self, other: "FurerNode", 
                     twisted: bool = False) -> bool:
        if self.raw_node == other.raw_node:
            return False
        con1 = other.raw_node in self.node_set
        con2 = self.raw_node in other.node_set
        return (con1 == con2) ^ twisted
    
    def __hash__(self) -> int:
        return hash((self.raw_node, frozenset(self.node_set)))

class MetaNode:
    def __init__(self):
        self.raw_node: int
        self.node_list: List[Set[int]] = []
    
    @staticmethod
    def from_adj(adj: AdjList, node: int) -> "MetaNode":
        meta_node = MetaNode()
        meta_node.raw_node = node
        meta_node.node_list = even_subsets(adj.adj_dict[node])
        return meta_node

    def __iter__(self):
        self.counter = 0
        return self
    
    def __next__(self) -> FurerNode:
        try:
            node = FurerNode(self.raw_node, self.node_list[self.counter])
        except IndexError:
            raise StopIteration
        self.counter += 1
        return node

class FurerGraph:
    def __init__(self, raw_graph: AdjList,
                 twist: List[Tuple[int, int]] = []):
        self.node_list: List[FurerNode] = sum(
            [list(MetaNode.from_adj(raw_graph, node))
             for node in range(raw_graph.num_nodes)], start=[]
        )
        self.node_feature: List[int] = sum(
            [len(list(MetaNode.from_adj(raw_graph, node))) * [node] 
             for node in range(raw_graph.num_nodes)], start=[]
        )
        self.twist = twist
        for node_pair in twist:
            assert raw_graph.has_edge(*node_pair)
    
    def to_sparse_adj(self) -> np.ndarray:
        adj: List[np.ndarray] = []
        for (i1, n1) in enumerate(self.node_list):
            for (i2, n2) in enumerate(self.node_list):
                if n1.is_connected(n2, 
                                   twisted=(n1.raw_node, n2.raw_node)
                                   in self.twist or
                                   (n2.raw_node, n1.raw_node)
                                   in self.twist):
                    adj.append(np.array([i1, i2], dtype=np.int64))
        return np.stack(adj).T
    
def get_furer_graph_pair(edge_index: np.ndarray) -> \
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert len(edge_index.shape) == 2 and\
           edge_index.shape[0] == 2 and\
           edge_index.shape[1] > 0
    adj = AdjList.from_sparse_adj(edge_index)
    G = FurerGraph(adj)
    H = FurerGraph(adj, [(edge_index[0, 0], edge_index[1, 0])])
    return (G.to_sparse_adj(), np.array(G.node_feature, dtype=np.float32).reshape(-1, 1),
            H.to_sparse_adj(), np.array(H.node_feature, dtype=np.float32).reshape(-1, 1),)
