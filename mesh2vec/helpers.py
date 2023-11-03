"""helper functions"""
from typing import OrderedDict, List, Tuple, Dict

import numpy as np
from scipy.sparse import csr_array, coo_array, eye, csr_matrix, lil_matrix, coo_matrix
import igraph as ip
from abc import ABC, abstractmethod
import time

"""
Use strategy to implement multiple adjacency finding algorithms
"""


class AbstractAdjacencyStrategy(ABC):
    @abstractmethod
    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Tuple[Dict[int, csr_array], Dict[int, csr_array]]:
        pass


class MatMulAdjacency(AbstractAdjacencyStrategy):
    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Tuple[Dict[int, csr_array], Dict[int, csr_array]]:
        """calc adjacencies for hyper nodes with matrix multiplication to find adjacent nodes for a given distance"""
        # vtx_connectivity: holds for each vtx all connected vtx

        timer = time.time()
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

        print("THIS TOOK: ", time.time() - timer)

        return adjacency_matrix_powers, adjacency_matrix_powers_exclusive


class BFSAdjacency(AbstractAdjacencyStrategy):
    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Tuple[Dict[int, csr_array], Dict[int, csr_array]]:
        # convert hyper edges to graph; hyper_edges: key -> node, values -> edges to directly connected nodes
        # as nodes are already integers, create graph from scratch
        # note: some nodes are integers written as strings -> conversion necessary
        timer = time.time()

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

        print("THIS TOOK: ", time.time() - timer)

        return adjacency_list_inclusive, adjacency_list_exclusive
