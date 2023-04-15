"""
batch.py - Correctly handle batching for triangle-level features.
"""
import torch
from pygmmpp.data import Data, Batch
from pygmmpp.data.collate import collate as my_collate
from typing import List


def collate(cls, data_list: List[Data]) -> Batch:
    """
    A revised edition of `pygmmpp.data.collate.collate()`, which now
    correctly handles triangle-level features. The interface is the
    same as `pygmmpp.data.collate.collate()`.

    To correctly use this `collate()` function, one should treat any
    triangle-level feature as a tensor-type attribute which adopts a
    `collate_type` of `'auto_collate'` (see documentation of 
    `pygmmpp.data.Data.__set_tensor_attr__()` function) and then calls
    into this `collate()` function.
    """
    # Call original `collate()` function to combine the
    # triangle-level feature into a single tensor
    # (without further increments)
    batch: Batch = my_collate(cls, data_list)

    # Now, any triangle-level feature requires additional care
    batch_level = batch.batch_level
    for key in batch.__dict__:
        if key.startswith('triangle_') and \
            '_slice' not in key:     # please name all 
            # triangle-level feature with 'triangle_i_j_k'
            # in which i, j, k are numbers, 
            # since we'll parse them here, thanks 
            _, i, j, k = key.split('_')
            
            # we (1) slice down proper piece of triangle-level
            # feature using its corresponding slicing vector
            # (2) add proper __inc__ tensor to it (NOT a simple
            # scalar adding now) using corresponding slicing
            # vectors of "edge_index{l}"
            slice_key = batch.__dict__[f'{key}_slice{batch_level}']
            slice_i, slice_j, slice_k = (
                batch.__dict__[f'{get_edge_slice(i)}{batch_level}'],
                batch.__dict__[f'{get_edge_slice(j)}{batch_level}'],
                batch.__dict__[f'{get_edge_slice(k)}{batch_level}']
            )
            num_samples = len(slice_key)
            for i in range(num_samples):
                left = slice_key[i]
                try:
                    right = slice_key[i+1]
                except IndexError:
                    right = batch.__dict__[key].shape[1]
                batch.__dict__[key][:, left:right] = batch.__dict__[
                    key][:, left:right] + torch.tensor(
                        [[slice_i[i]], [slice_j[i]], [slice_k[i]]], dtype=torch.int64
                    )
        elif key.startswith('inverse_edge'):
            _, _, k = key.split('_')

            slice_k = batch.__dict__[f'{get_edge_slice(k)}{batch_level}']
            num_samples = len(slice_k)
            for i in range(num_samples):
                left = slice_k[i]
                try:
                    right = slice_k[i+1]
                except IndexError:
                    right = batch.__dict__[key].shape[0]
                batch.__dict__[key][left:right] = batch.__dict__[
                    key][left:right] + left
    return batch

"""
NOTE: Here we only re-implemented the `collate()` method but without the
corresponding `separate()` method. This is because the `collate()` method
defined here is aimed to be **ONLY** called when loading batches of data.
Therefore, one must call the library-given `collate()` method when
constructing a `Dataset` from raw data. In practice this means calling the
high-level functions (e.g. `Batch.from_data_list()`) and not touching down
to the implementation details. However, when using a `DataLoader` to load
data into batches, one must pass the `collate()` fucntion defined here as
an argument of `DataLoader()` in order to produce expected result for a 
batch. 
"""

# A small helper function to find the correct "edge_index" name
def get_edge_index(k: str) -> str:
    return f'edge_index{k}' if k != '1' else 'edge_index'
    #return f'edge_index{k}'

# A small helper function to find the correct "edge_slice" name
def get_edge_slice(k: str) -> str:
    return f'{get_edge_index(k)}_slice' if k != '1' else 'edge_slice'
    #return f'{get_edge_index(k)}_slice'