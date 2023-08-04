"""
triangle.py - Generate triangle-level feature for graphs.
"""
import torch
import numpy as np

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
    # version 1: faster for smaller graphs but can't scale to larger graphs
    # mask_m_edge = (node_mat.unsqueeze(-1) == edge_index_m_1d).any(-1)
    # version 2: scale to larger graphs
    mask_m_edge = torch.isin(node_mat, edge_index_m_1d)

    # Final mask which selects the triangles needed
    mask = mask_share_start * mask_m_edge

    # Get index for m_edges
    # version 1: faster
    index_m_edge = torch.where(node_mat[mask].unsqueeze(1)
                               == edge_index_m_1d.unsqueeze(0))[1]
    # version 2: use if memory is very limited (e.g. ogbg-ppa)
    # ind = torch.argsort(edge_index_m_1d)
    # edge_index_m_1d_sorted = edge_index_m_1d[ind]
    # index_m_edge = ind[torch.searchsorted(edge_index_m_1d_sorted, node_mat[mask])]
    
    # Get index for l_edges and k_edges
    index_l_edge, index_k_edge = torch.where(mask)
    
    return torch.stack([index_l_edge, index_k_edge, index_m_edge]).cpu()

# def get_lkm_triangles(edge_index_l: torch.LongTensor,
#                       edge_index_k: torch.LongTensor,
#                       edge_index_m: torch.LongTensor,
#                       num_nodes: int) \
#     -> torch.LongTensor:
#     """
#     Generate a list of triangles whose three edge lengths are
#     `(l, k, m)` within a graph. Each triangle is represented by
#     a 3-tuple of node pairs `(h, i, j)` where `h, i, j` are indices
#     into `edge_index` tensor corresponding to node pairs with
#     distances `l, k, m`.

#     This operation only supports CPU backends currently.

#     Args:

#     edge_index_l, edge_index_k, edge_index_m: `edge_index` tensors
#     corresponding to distance `l, k, m` respectively
#     """
#     triangle_list = []

#     # now we loop over every l-edge in the graph
#     for l_edge in range(edge_index_l.shape[1]):
#         i = edge_index_l[0, l_edge]
#         j = edge_index_l[1, l_edge]

#         # first, we calculate "what are the k-hop edges
#         # starting from node i"
#         # we label indices into such edges as `idx_i`
#         idx_i, = torch.where(edge_index_k[0] == i)

#         # second, we calculate "what are the m-hop edges
#         # ending at node j"
#         # we label indices into such edges as `idx_j`
#         idx_j, = torch.where(edge_index_m[1] == j)

#         # finally, we find the intersections of nodes
#         # belonging to 
#         # (a) edge_index_k[1][idx_i]
#         # (b) edge_index_m[0][idx_j]
#         # and their indices back.

#         # This step is only available on CPU currently.
#         _, ind_back_i, ind_back_j = np.intersect1d(
#             edge_index_k[1][idx_i].numpy(),
#             edge_index_m[0][idx_j].numpy(),
#             assume_unique=True,
#             return_indices=True
#         )
#         triangle_list.append(torch.stack([
#             torch.full_like(torch.from_numpy(_), 
#                             l_edge, dtype=torch.int64),
#             idx_i[ind_back_i], idx_j[ind_back_j]],
#             dim=0
#         ))
#     return torch.cat(triangle_list, dim=1)

# def get_lkm_triangles(edge_index_l: torch.LongTensor,
#                       edge_index_k: torch.LongTensor,
#                       edge_index_m: torch.LongTensor,
#                       num_nodes: int) \
#     -> torch.LongTensor:
#     """
#     Generate a list of triangles whose three edge lengths are
#     `(l, k, m)` within a graph. Each triangle is represented by
#     a 3-tuple of node pairs `(h, i, j)` where `h, i, j` are indices
#     into `edge_index` tensor corresponding to node pairs with
#     distances `l, k, m`.

#     Args:

#     edge_index_l, edge_index_k, edge_index_m: `edge_index` tensors
#     corresponding to distance `l, k, m` respectively

#     num_nodes: number of nodes
#     """
#     triangle_list = []
#     # Translate edge_index_m into a 1D-tensor
#     edge_index_m_1d = edge_index_m[1] * num_nodes + edge_index_m[0]
    
#     # Loop over node i
#     for i in range(num_nodes):
#         l_edges_begins_at_i = torch.where(edge_index_l[0] == i)[0]
#         k_edges_begins_at_i = torch.where(edge_index_k[0] == i)[0]

#         # We select nodes that are at distance l from i and nodes
#         # that are at distance k from i, and then combine them into
#         # a matrix which gives every node pair a unique value
#         node_mat = edge_index_l[1][l_edges_begins_at_i].unsqueeze(
#             1) * num_nodes + edge_index_k[1][k_edges_begins_at_i]
        
#         # Copy down indices from other two edges
#         mask = (node_mat.unsqueeze(-1) == edge_index_m_1d).any(-1)
#             # a faster version for torch.isin(node_mat, edge_index_m_1d)
        
#         # We store the indices into edge_index_m_1d tensor of node 
#         # pairs (i, j) within matrix node_mat in `indices`, which would
#         # be used for constructing triangle lists
#         indices = torch.where(node_mat.flatten().unsqueeze(1) 
#                               == edge_index_m_1d.unsqueeze(0))[1]

#         triangle_list.append(torch.cat([torch.stack([
#             l_edges_begins_at_i.unsqueeze(1).broadcast_to(node_mat.shape),
#             k_edges_begins_at_i.broadcast_to(node_mat.shape)
#         ])[:, mask], indices.unsqueeze(0)]))
#     return torch.cat(triangle_list, dim=1)