import numpy as np
from torch import LongTensor, cat, stack
from typing import Dict
from IEProtLib.py_utils.py_mol.PyProtein import PyProtein

def read_file(protein: PyProtein, filename: str) -> Dict:
    protein.load_hdf5(filename)
    return {'AminoChainIDs': protein.aminoChainIds_,
            'AminoPos': protein.aminoPos_,
            'AminoType': protein.aminoType_,
            'AminoNeighbors': protein.aminoNeighs_,
            'AminoNeighborsSIndices': protein.aminoNeighsSIndices_,
            'AminoNeighborsHB': protein.aminoNeighsHB_,
            'AminoNeighborsSIndicesHB': protein.aminoNeighsSIndicesHB_,
            }

def get_num_nodes(file_dict: Dict) -> int:
    return file_dict['AminoPos'].shape[1]

def get_edge_index(file_dict: Dict, includeHB: bool) -> np.ndarray:
    return file_dict['AminoNeighborsHB'] if includeHB else file_dict['AminoNeighbors']

def get_amino_type(file_dict: Dict) -> np.ndarray:
    return file_dict['AminoType'].reshape((-1, 1))

def get_amino_pos(file_dict: Dict) -> np.ndarray:
    return file_dict['AminoPos'].reshape((-1, 3))

def to_undirected(edge_index: LongTensor, num_nodes: int):
    edge_index_1d = edge_index[0] * num_nodes + edge_index[1]
    edge_index_1d_rev = edge_index[1] * num_nodes + edge_index[0]
    new_edge_index = cat([edge_index_1d, edge_index_1d_rev], dim=0).unique()
    return stack([new_edge_index // num_nodes, new_edge_index % num_nodes], dim=0)
