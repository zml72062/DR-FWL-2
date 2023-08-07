from typing import Set, List
from itertools import combinations

def even_subsets(s: Set[int]) -> List[Set[int]]:
    num_elems = len(s)
    subsets = []
    for n in range(0, num_elems + 1, 2):
        subsets += list(map(set, combinations(s, n)))
    return subsets

def odd_subsets(s: Set[int]) -> List[Set[int]]:
    num_elems = len(s)
    subsets = []
    for n in range(1, num_elems + 1, 2):
        subsets += list(map(set, combinations(s, n)))
    return subsets
