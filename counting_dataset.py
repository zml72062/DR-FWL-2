from pygmmpp.datasets import CountDataset
## Must run `make` under directory "./software/cycle_count" first
## otherwise can't import
import cycle_count

target_dict = {
    '3-cycle': lambda i, n, e: cycle_count.count_cycles(i, n, e, 3),
    '4-cycle': lambda i, n, e: cycle_count.count_cycles(i, n, e, 4),
    '5-cycle': lambda i, n, e: cycle_count.count_cycles(i, n, e, 5),
    '6-cycle': lambda i, n, e: cycle_count.count_cycles(i, n, e, 6),
    '7-cycle': lambda i, n, e: cycle_count.count_cycles(i, n, e, 7),
    '4-clique': cycle_count.count_4_cliques,
    'tailed-triangle': cycle_count.count_tailed_triangles,
    'chordal-cycle': cycle_count.count_chordal_cycles,
    'triangle-rectangle': cycle_count.count_triangle_rectangles,
    '4-path': lambda i, n, e: cycle_count.count_paths(i, n, e, 4).sum(axis=0)
}

def get_count_dataset(root: str, target: str, split: str = 'train', **kwargs):
    return CountDataset(root, target_dict[target], target, split, **kwargs)
