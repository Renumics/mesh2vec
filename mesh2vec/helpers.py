"""helper functions"""
from typing import OrderedDict, List, Tuple, Dict

import numpy as np
from scipy.sparse import csr_array, coo_array, eye


def calc_adjacencies(
    hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
) -> Tuple[Dict[int, csr_array], Dict[int, csr_array]]:
    """calc adjacencies for hyper nodes"""
    # vtx_connectivity: holds for each vtx all connected vtx
    adjacency_list = []
    for vtxs in hyper_edges_idx.values():
        for vtx_a in vtxs:
            for vtx_b in vtxs:
                adjacency_list.append([vtx_a, vtx_b])
    adjacency_list_np = np.unique(np.array(adjacency_list), axis=0)

    # adjacency matrix
    n_vtx = max(max(v) for v in hyper_edges_idx.values()) + 1
    data = np.array([1] * len(adjacency_list_np))
    adjacency_matrix = coo_array(
        (data, (adjacency_list_np[:, 1], adjacency_list_np[:, 0])), shape=(n_vtx, n_vtx)
    ).tocsr()

    # neighbors
    adjacency_matrix_powers = {1: adjacency_matrix}
    adjacency_matrix_powers_exclusive = {1: adjacency_matrix - csr_array(eye(n_vtx, dtype=int))}
    for i in range(2, max_distance + 1):
        adjacency_matrix_powers[i] = adjacency_matrix_powers[i - 1] @ adjacency_matrix
        adjacency_matrix_powers[i].data = np.array([1] * adjacency_matrix_powers[i].nnz)
        exclusive = (adjacency_matrix_powers[i] - adjacency_matrix_powers[i - 1]).tocsr()
        adjacency_matrix_powers_exclusive[i] = exclusive
    return adjacency_matrix_powers, adjacency_matrix_powers_exclusive
