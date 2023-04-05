"""
triangle.py - Generate triangle-level feature for graphs.
"""
import torch

def get_lkm_triangles(edge_index_l: torch.LongTensor,
                      edge_index_k: torch.LongTensor,
                      edge_index_m: torch.LongTensor,
                      num_nodes: int) \
    -> torch.LongTensor:
    """
    Generate a list of triangles whose three edge lengths are
    `(l, k, m)` within a graph. Each triangle is represented by
    a 3-tuple of node pairs `(h, i, j)` where `h, i, j` are indices
    into `edge_index` tensor corresponding to node pairs with
    distances `l, k, m`.

    Args:

    edge_index_l, edge_index_k, edge_index_m: `edge_index` tensors
    corresponding to distance `l, k, m` respectively

    num_nodes: number of nodes
    """
    # Get a boolean mask of "where the l_edges and k_edges share
    # starting point"
    mask_share_start = edge_index_l[0].unsqueeze(
        1) == edge_index_k[0].unsqueeze(0)
    
    # Construct unique integer identifier for every node pair in the
    # matrix: meshgrid(edge_index_l[1], edge_index_k[1])
    node_mat = edge_index_l[1].unsqueeze(
        1) * num_nodes + edge_index_k[1]
    
    # Get a boolean mask of "whether the node pairs are bonded by an
    # m_edge"
    edge_index_m_1d = edge_index_m[1] * num_nodes + edge_index_m[0]
    mask_m_edge = (node_mat.unsqueeze(-1) == edge_index_m_1d).any(-1)

    # Final mask which selects the triangles needed
    mask = mask_share_start * mask_m_edge

    # Get index for m_edges
    index_m_edge = torch.where(node_mat[mask].unsqueeze(1)
                               == edge_index_m_1d.unsqueeze(0))[1]
    # Get index for l_edges and k_edges
    index_l_edge, index_k_edge = torch.where(mask)
    
    return torch.stack([index_l_edge, index_k_edge, index_m_edge])