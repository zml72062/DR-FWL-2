import torch
import numpy as np
import ctypes
import os
from typing import Union

################### Definition for C interfaces ################

cycle_module = ctypes.CDLL(os.path.join(os.path.split(__file__)[0], 
                                        'find_cycle.so'))

c_count_cycles = cycle_module.graph_count_cycles
c_count_cycles.argtypes = [ctypes.POINTER(ctypes.c_long),
                           ctypes.c_long,
                           ctypes.c_long,
                           ctypes.c_long, 
                           ctypes.POINTER(ctypes.c_long)]

c_count_paths = cycle_module.graph_count_paths
c_count_paths.argtypes = [ctypes.POINTER(ctypes.c_long),
                          ctypes.c_long,
                          ctypes.c_long,
                          ctypes.c_long, 
                          ctypes.POINTER(ctypes.c_long)]

c_count_substruct = cycle_module.graph_count_substruct
c_count_substruct.argtypes = [ctypes.POINTER(ctypes.c_long),
                              ctypes.c_long,
                              ctypes.c_long,
                              ctypes.POINTER(ctypes.c_long),
                              ctypes.c_char_p]

#################################################################

def count_cycles(edge_index: Union[np.ndarray, torch.LongTensor],
                 num_nodes: int, num_edges: int, k: int) -> np.ndarray:
    """
    Returns node-level k-cycle count in shape (num_nodes, )
    """
    count = np.zeros((num_nodes, ), dtype=np.int64)
    c_count_cycles(np.ctypeslib.as_ctypes(
                        edge_index.reshape(-1, 1).squeeze().numpy()
                        if isinstance(edge_index, torch.Tensor) else
                        edge_index.reshape(-1, 1).squeeze()), 
                   ctypes.c_long(num_nodes), 
                   ctypes.c_long(num_edges), 
                   ctypes.c_long(k),
                   np.ctypeslib.as_ctypes(count))
    return count

def count_paths(edge_index: Union[np.ndarray, torch.LongTensor],
                num_nodes: int, num_edges: int, k: int) -> np.ndarray:
    """
    Returns node-pair-level k-path count in shape (num_nodes, num_nodes)
    """
    count = np.zeros((num_nodes * num_nodes, ), dtype=np.int64)
    c_count_paths(np.ctypeslib.as_ctypes(
                        edge_index.reshape(-1, 1).squeeze().numpy()
                        if isinstance(edge_index, torch.Tensor) else
                        edge_index.reshape(-1, 1).squeeze()), 
                   ctypes.c_long(num_nodes), 
                   ctypes.c_long(num_edges), 
                   ctypes.c_long(k),
                   np.ctypeslib.as_ctypes(count))
    return count.reshape((num_nodes, num_nodes))

def node_level_count_paths(edge_index: Union[np.ndarray, torch.LongTensor],
                           num_nodes: int, num_edges: int, k: int) -> np.ndarray:
    """
    Returns node-level k-path count in shape (num_nodes, )
    """
    return count_paths(edge_index, num_nodes, num_edges, k).sum(axis=1)

def count_4_cliques(edge_index: Union[np.ndarray, torch.LongTensor],
                    num_nodes: int, num_edges: int) -> np.ndarray:
    """
    Returns node-level 4-clique count in shape (num_nodes, )
    """
    count = np.zeros((num_nodes, ), dtype=np.int64)
    c_count_substruct(np.ctypeslib.as_ctypes(
                        edge_index.reshape(-1, 1).squeeze().numpy()
                        if isinstance(edge_index, torch.Tensor) else
                        edge_index.reshape(-1, 1).squeeze()), 
                   ctypes.c_long(num_nodes), 
                   ctypes.c_long(num_edges), 
                   np.ctypeslib.as_ctypes(count), b'cl')
    return count

def count_chordal_cycles(edge_index: Union[np.ndarray, torch.LongTensor],
                         num_nodes: int, num_edges: int) -> np.ndarray:
    """
    Returns node-level chordal-cycle count in shape (num_nodes, )
    """
    count = np.zeros((num_nodes, ), dtype=np.int64)
    c_count_substruct(np.ctypeslib.as_ctypes(
                        edge_index.reshape(-1, 1).squeeze().numpy()
                        if isinstance(edge_index, torch.Tensor) else
                        edge_index.reshape(-1, 1).squeeze()), 
                   ctypes.c_long(num_nodes), 
                   ctypes.c_long(num_edges), 
                   np.ctypeslib.as_ctypes(count), b'cc')
    return count
    
def count_triangle_rectangles(edge_index: Union[np.ndarray, torch.LongTensor],
                              num_nodes: int, num_edges: int) -> np.ndarray:
    """
    Returns node-level triangle-rectangle count in shape (num_nodes, )
    """
    count = np.zeros((num_nodes, ), dtype=np.int64)
    c_count_substruct(np.ctypeslib.as_ctypes(
                        edge_index.reshape(-1, 1).squeeze().numpy()
                        if isinstance(edge_index, torch.Tensor) else
                        edge_index.reshape(-1, 1).squeeze()), 
                   ctypes.c_long(num_nodes), 
                   ctypes.c_long(num_edges), 
                   np.ctypeslib.as_ctypes(count), b'tr')
    return count

def count_tailed_triangles(edge_index: Union[np.ndarray, torch.LongTensor],
                           num_nodes: int, num_edges: int) -> np.ndarray:
    """
    Returns node-level tailed-triangle count in shape (num_nodes, )
    """
    count = np.zeros((num_nodes, ), dtype=np.int64)
    c_count_substruct(np.ctypeslib.as_ctypes(
                        edge_index.reshape(-1, 1).squeeze().numpy()
                        if isinstance(edge_index, torch.Tensor) else
                        edge_index.reshape(-1, 1).squeeze()), 
                   ctypes.c_long(num_nodes), 
                   ctypes.c_long(num_edges), 
                   np.ctypeslib.as_ctypes(count), b'tt')
    return count

# Utility function
def get_name(func):
    return str(func).split(' ')[1]