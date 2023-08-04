from .IEProtLib.py_utils.py_mol.PyProtein import PyProtein
from .IEProtLib.py_utils.py_mol.PyPeriodicTable import PyPeriodicTable
from . import utils
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import sys
from tqdm import tqdm
import torch
from typing import Optional, Callable, List
from .count import *

def get_split(split: str, root: str):
    assert split in ['training', 'validation', 'test_family', 'test_fold', 'test_superfamily'], \
    "Invalid split name!"
    with open(osp.join(root, 'raw', 'HomologyTAPE', f'{split}.txt')) as f:
        return [l.rstrip().split('\t')[0] for l in f.readlines()]

class PygHomologyTAPEDatasetWithCount(InMemoryDataset):
    def __init__(self, root: str, split: str, includeHB: bool = False,
                 to_undirected: bool = True,
                 use_amino_type: bool = False,
                 use_amino_pos: bool = False,
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        """
        Args:

        root (str): Root directory of the dataset \\
        split (str): Dataset split (training/validation/test_family/test_fold/test_superfamily) \\
        includeHB (bool): Whether to include hydrogen bond in adjacency matrix,
        default to False \\ 
        to_undirected (bool): Whether to convert graph into undirected by adding
        inverse edges, default to True \\
        use_amino_type (bool): Whether to use amino acid type as additional node
        feature, default to False \\
        use_amino_pos (bool): Whether to use amino acid position as additional
        node feature, default to False
        """
        self.split = split
        self.includeHB = includeHB
        self.to_undirected = to_undirected
        self.use_amino_type = use_amino_type
        self.use_amino_pos = use_amino_pos
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [osp.join('HomologyTAPE', self.split, f"{file}.hdf5") for file in get_split(self.split, self.root)
                ] + [osp.join('HomologyTAPE', f"{self.split}.txt")]
    
    @property
    def processed_file_names(self):
        return f'{self.split}.pt'

    def download(self):
        pass

    def process(self):
        periodictable = PyPeriodicTable()
        protein = PyProtein(periodictable)
        split = get_split(self.split, self.root)

        print("Converting hdf5 files to PyG objects...", file=sys.stderr)
        data_list: List[Data] = []
        for file in tqdm(split):
            file_dict = utils.read_file(protein, osp.join(self.root, 'raw', 'HomologyTAPE', self.split, f'{file}.hdf5'))
            num_nodes = utils.get_num_nodes(file_dict)
            edge_index = torch.from_numpy(utils.get_edge_index(file_dict, self.includeHB)).t().to(torch.long)
            if self.to_undirected:
                edge_index = utils.to_undirected(edge_index, num_nodes)
            x = torch.ones((num_nodes, 1))
            targets = ['3-cycle', '4-cycle', '5-cycle', '6-cycle', '4-path']
            target_list = []
            for target in targets:
                target_list.append(torch.from_numpy(target_dict[target](edge_index, num_nodes, edge_index.shape[1])).unsqueeze(1))
            y = torch.cat(target_list, dim=1)
            if self.use_amino_pos:
                amino_pos = utils.get_amino_pos(file_dict)
                x = torch.cat([x, amino_pos], dim=1)
            if self.use_amino_type:
                amino_type = utils.get_amino_type(file_dict)
                x = torch.cat([x, amino_type], dim=1)
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        
        print("Saving...", file=sys.stderr)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
