"""
preprocess.py - Preprocessing for 2-DR-FWL(2)
"""
import torch
from pygmmpp.utils.neighbor import k_hop_edge_index
from pygmmpp.data import Data
from .triangle import get_lkm_triangles
from .batch import get_edge_index
from .inverse_edge import inverse_edge
from typing import Callable, Dict, Optional
from pygmmpp.utils import compose



def generate_k_hop_neighbor(k: int) -> Callable:
    """
    Generate a preprocessing function that injects up to k-hop
    neighbor information into a `Data` object, using format
    like `edge_index`.

    Args:

    k (int) --- the maximum hop number
    """
    def inject_k_hop_neighbor(data: Data) -> Data:
        edge_index_dict: Dict[str, torch.Tensor] = k_hop_edge_index(
            k, data.edge_index, data.num_nodes
        )
        for (key, val) in edge_index_dict.items():
            data.__set_tensor_attr__(key, val, 'edge_index', slicing=True)
        return data
    return inject_k_hop_neighbor

def generate_k_hop_neighbor_feature(k: int,
                                    use_node_feature: bool = True) -> Callable:
    """
    Generate a preprocessing function that gives each pair of nodes
    within k-hop a feature vector, using format like `edge_attr`.

    Args:

    k (int) --- the maximum hop number

    use_node_feature (bool) --- if `True`, will use node feature X to
    generate pair-level feature: feature[i, j] = X[i] * X[j], default
    `True`
    """
    def inject_k_hop_attr(data: Data) -> Data:
        for hop in range(2, k+1):
            if use_node_feature:
                assert hasattr(data, 'x'), "Node feature absent!"
                src, tgt = data.__dict__['edge_index'+str(hop)]
                x = data.x
                feature = x[src] + x[tgt]
            else:
                feature = torch.ones(
                    (data.__dict__['edge_index'+str(hop)].shape[1], 1)
                )
            data.__set_tensor_attr__(
                'edge_attr'+str(hop), 
                feature,
                collate_type='auto_collate',
                cat_dim=0, slicing=True, use_slice='edge_index'+str(hop)
            )
        if not hasattr(data, 'edge_attr'):
            if use_node_feature:
                assert hasattr(data, 'x'), "Node feature absent!"
                src, tgt = data.edge_index
                x = data.x
                feature = x[src] + x[tgt]
            else:
                feature = torch.ones(
                    (data.edge_index.shape[1], 1)
                )
            data.__set_tensor_attr__('edge_attr', feature, 
                                     collate_type='edge_feature')
        elif use_node_feature:
            assert hasattr(data, 'x'), "Node feature absent!"
            src, tgt = data.edge_index
            x = data.x
            feature = x[src] + x[tgt]
            data.edge_attr = torch.cat([data.edge_attr, feature], dim=1)
        return data
    return inject_k_hop_attr

def generate_lkm_triangle(l: int, k: int, m: int) -> Callable:
    """
    Generate a preprocessing function that records every triangle 
    whose three edge lengths are (l, k, m) using three indices into
    the tensors in which the three "edges" lie.

    Args:

    l, k, m (int) --- the edge lengths of the triangle
    """
    def inject_lkm_triangle(data: Data) -> Data:
        data.__set_tensor_attr__(f'triangle_{l}_{k}_{m}',
                                 get_lkm_triangles(
                                    data.__dict__[get_edge_index(f"{l}")],
                                    data.__dict__[get_edge_index(f"{k}")],
                                    data.__dict__[get_edge_index(f"{m}")],
                                    data.num_nodes
                                 ),
                                 'auto_collate', cat_dim=1, slicing=True
        )
        return data
    return inject_lkm_triangle

def generate_inverse_edge(l: int) -> Callable:
    """
    Generate the permutation that maps the edge (u, v) to its inverse
    edge (v, u). The edges are with hop <= l.

    Args:

    l (int) --- the maximum hop
    """
    def inject_inverse_edge_permutation(data: Data) -> Data:
        num_nodes = data.num_nodes
        data.__set_tensor_attr__(f"inverse_edge_1",
                                 inverse_edge(
                                    data.edge_index, num_nodes
                                 ),
                                 'edge_feature')
        for k in range(2, l+1):
            data.__set_tensor_attr__(f"inverse_edge_{k}",
                                     inverse_edge(
                                        data.__dict__[f'edge_index{k}'], num_nodes
                                     ),
                                     'auto_collate', 0, True, f'edge_index{k}')
        return data
    return inject_inverse_edge_permutation

def add_root_edges(data: Data) -> Data:
    num_nodes = data.num_nodes
    index = list(range(num_nodes))
    edge_index0 = torch.tensor([index, index]).long()
    data.__set_tensor_attr__("edge_index0", edge_index0, 'edge_index', slicing=True)
    return data


def drfwl2_transform():
    pretransform = compose(
        [generate_k_hop_neighbor(2),
         add_root_edges,
         #generate_k_hop_neighbor_feature(2, False),
         generate_lkm_triangle(0, 1, 1),
         generate_lkm_triangle(1, 1, 1),
         generate_lkm_triangle(1, 1, 2),
         generate_lkm_triangle(1, 2, 2),
         generate_lkm_triangle(2, 2, 2),
         generate_inverse_edge(2)]
    )
    return pretransform



###### As a comparison, we implemented NGNN preprocessing as below

def ngnn_transform(k: int):
    from pygmmpp.utils import k_hop_subgraph
    from pygmmpp.data import Batch
    def subgraph_batch(data: Data) -> Batch:
        data_list = []
        for node in range(data.num_nodes):
            subset, edge_index, _, edge_mask = k_hop_subgraph(
                node, k, data.edge_index, data.num_nodes, relabel_nodes=True)
            kwargs = {'x': data.x[subset],
                      'edge_index': edge_index,
                      'edge_attr': data.edge_attr[edge_mask]
                      if data.edge_attr is not None else None,
                      'y': data.y}
            data_list.append(Data(**kwargs))
        return Batch.from_data_list(data_list)
    return subgraph_batch

