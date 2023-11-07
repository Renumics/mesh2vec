"""helper functions"""
from typing import OrderedDict, List, Tuple, Dict

import numpy as np
from scipy.sparse import csr_array, coo_array, eye
import igraph as ip
from abc import ABC, abstractmethod

"""
Use strategy to implement multiple adjacency finding algorithms
"""


class AbstractAdjacencyStrategy(ABC):
    def __init__(self):
        self.__adjacency_matrix_powers = None
        self.__adjacency_matrix_powers_exclusive = None
    
    def set_matrices(self, matrix_powers, matrix_powers_exclusive):
        self.__adjacency_matrix_powers = matrix_powers
        self.__adjacency_matrix_powers_exclusive = matrix_powers_exclusive

    @abstractmethod
    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Tuple[Dict[int, csr_array], Dict[int, csr_array]]:
        pass

    @abstractmethod
    def get_neighbors_exclusive(self, distance: int, vtx_id: int) -> List[int]:
        pass


class MatMulAdjacency(AbstractAdjacencyStrategy):
    def __init__(self):
        super().__init__()

    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Tuple[Dict[int, csr_array], Dict[int, csr_array]]:
        """calc adjacencies for hyper nodes with matrix multiplication to find adjacent nodes for a given distance"""
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
        adjacency_matrix_powers_exclusive = {
            1: adjacency_matrix - csr_array(eye(n_vtx, dtype=int))
        }

        for i in range(2, max_distance + 1):
            adjacency_matrix_powers[i] = adjacency_matrix_powers[i - 1] @ adjacency_matrix
            adjacency_matrix_powers[i].data = np.array([1] * adjacency_matrix_powers[i].nnz)
            exclusive = (adjacency_matrix_powers[i] - adjacency_matrix_powers[i - 1]).tocsr()
            adjacency_matrix_powers_exclusive[i] = exclusive

        self.set_matrices(adjacency_matrix_powers, adjacency_matrix_powers_exclusive)
        return adjacency_matrix_powers, adjacency_matrix_powers_exclusive
    
    def get_neighbors_exclusive(self, distance: int, vtx_id: int) -> List[int]:
        return self.__adjacency_matrix_powers_exclusive[distance][vtx_id]
    
    def __str__(self) -> str:
        return "MatMulAdjacency"


class BFSAdjacency(AbstractAdjacencyStrategy):
    def __init__(self):
        super().__init__()

    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Tuple[Dict[int, csr_array], Dict[int, csr_array]]:

        adjacency_list = []
        for vtxs in hyper_edges_idx.values():
            for vtx_a in vtxs:
                for vtx_b in vtxs:
                    adjacency_list.append([vtx_a, vtx_b])
        adjacency_list_np = np.unique(np.array(adjacency_list), axis=0)

        graph = ip.Graph(adjacency_list_np)

        adj_list = graph.get_adjlist()
        adjacency_list_exclusive = {1: adj_list}
        adjacency_list_inclusive = {1: adj_list}

        ####################################################
        # General idea: BFS for each vertex for each depth #
        ####################################################
        for depth in range(2, max_distance + 1):
            exclusive_neighborhood = graph.neighborhood(vertices=None, mindist=depth, order=depth)
            inclusive_neighborhood = graph.neighborhood(vertices=None, order=depth)
            adjacency_list_exclusive[depth] = exclusive_neighborhood
            adjacency_list_inclusive[depth] = inclusive_neighborhood

        self.set_matrices(adjacency_list_inclusive, adjacency_list_exclusive)
        return adjacency_list_inclusive, adjacency_list_exclusive
    
    def get_neighbors_exclusive(self, distance: int, vtx_id: int) -> List[int]:
        return self.__adjacency_matrix_powers_exclusive[distance][vtx_id]
    
    def __str__(self) -> str:
        return "BFSAdjacency"
