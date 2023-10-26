import networkx as nx
import numpy as np
import torch
from pygmmpp.data import Dataset, Batch, Data
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import os
from tqdm import tqdm

torch_geometric.seed_everything(2022)

def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


def add_x(data):
    data.x = torch.ones([data.num_nodes, 1]).to(torch.long)
    return data

class BRECDataset(Dataset):
    def __init__(
        self,
        name="no_param",
        root="BREC/Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_batch = torch.load(self.processed_paths[0])
        self.indices = torch.arange(len(self.data_batch))

    @property
    def processed_dir(self):
        name = 'processed'
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_file_names(self):
        return ["brec_v3.pt"]
    
    def download(self):
        pass

    def process(self):
        data_list = np.load(self.raw_paths[0], allow_pickle=True)

        processed_list = []
        for i, data in enumerate(tqdm(data_list)):
            data = add_x(graph6_to_pyg(data))
            data = Data(x=data.x, edge_index=data.edge_index)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            processed_list.append(data)

        torch.save(Batch.from_data_list(processed_list), self.processed_paths[0])

