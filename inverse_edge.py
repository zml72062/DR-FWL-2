import torch

def inverse_edge(edge_index: torch.LongTensor,
                 num_nodes: int) -> torch.LongTensor:
    """
    Generate a 1D permutation vector that maps an edge to its 
    inverse edge in `edge_index`, i.e. the function returns a
    tensor p such that 
    ```
    edge_index[0][p], edge_index[1][p] = edge_index[1], edge_index[0]
    ```
    """
    row_first = edge_index[0] * num_nodes + edge_index[1]
    col_first = edge_index[1] * num_nodes + edge_index[0]

    row_sort = torch.argsort(row_first)
    col_sort = torch.argsort(col_first)

    col_inv = torch.empty_like(col_first)
    col_inv[col_sort] = torch.arange(col_first.shape[0])

    return row_sort[col_inv]
