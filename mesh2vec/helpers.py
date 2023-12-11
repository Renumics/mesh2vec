"""helper functions"""
from typing import OrderedDict, List, Dict
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_array, coo_array, eye

# pylint: disable=invalid-name
class AbstractAdjacencyStrategy(ABC):
    # pylint: disable=too-few-public-methods
    """
    Abstract class for adjacency finding strategies
    """

    @abstractmethod
    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Dict[int, List[List[int]]]:
        """
        calculate adjacencies for hyper nodes with a given maximum distance
        Args:
            hyper_edges_idx: OrderedDict of hyper nodes with their indices
            max_distance: maximum distance to find adjacencies for
        Returns:
            dict of lists of lists (distance, vertex, neighbors)
        """


def _hyper_edges_to_adj_np(hyper_edges_idx):
    adjacency_list = []
    for vtxs in hyper_edges_idx.values():
        for vtx_a in vtxs:
            for vtx_b in vtxs:
                adjacency_list.append([vtx_a, vtx_b])
    adjacency_list_np = np.unique(np.array(adjacency_list), axis=0)
    return adjacency_list_np


def _hyper_edges_to_adj_list(vtx_count, hyper_edges_idx, include_self=False):
    adjacency_list = [[] for _ in range(vtx_count)]
    for vtxs in hyper_edges_idx.values():
        for vtx_a in vtxs:
            for vtx_b in vtxs:
                if include_self or not vtx_a == vtx_b:
                    adjacency_list[vtx_a].append(vtx_b)
                    adjacency_list[vtx_b].append(vtx_a)
    adjacency_list = [list(set(adjacency)) for adjacency in adjacency_list]
    return adjacency_list


class MatMulAdjacency(AbstractAdjacencyStrategy):
    # pylint: disable=too-few-public-methods
    """calc adjacencies using matrix multiplication"""

    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Dict[int, List[List[int]]]:
        """calc adjacencies using matrix multiplication"""
        # vtx_connectivity: holds for each vtx all connected vtx

        adjacency_list_np = _hyper_edges_to_adj_np(hyper_edges_idx)

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

        ## neighborhoods key=distance, value=list of neighbors by index
        neighborhoods = {}
        for dist in range(1, max_distance + 1):
            neighborhoods[dist] = [
                adjacency_matrix_powers_exclusive[dist][[i], :].indices.tolist()
                for i in range(n_vtx)
            ]
        return neighborhoods


class PurePythonBFS(AbstractAdjacencyStrategy):
    # pylint: disable=too-few-public-methods
    """calc adjacencies using BFS in pure python"""

    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Dict[int, List[List[int]]]:
        """calc adjacencies using BFS in pure python"""
        # values in .values(): indices list -> maximum index + 1 is length of unique vertices
        vtx_count = max(vtx_a for vtxs in hyper_edges_idx.values() for vtx_a in vtxs) + 1
        adjacency_list = _hyper_edges_to_adj_list(vtx_count, hyper_edges_idx)

        # each vertex has a list of lists where the index of the list corresponds
        #   to the depth of its contained neighbors
        # e.g.: neighbor 300 at depth 2, neighbor 400 at depth 1 with
        #    max_distance 3 -> [[], [400], [300], []]
        neighbors_at_depth = {
            dist: [[] for vertex in range(vtx_count)] for dist in range(max_distance + 1)
        }

        # for each vertex, do bfs separately
        for start_vertex in range(vtx_count):
            # Initialize the queue for BFS: (vertex, depth)
            queue = deque([(start_vertex, 0)])

            # Use a set to keep track of visited vertices
            visited = set([start_vertex])

            while queue:
                # depth is saved in queue -> no tracking in loop
                vertex, depth = queue.popleft()

                # do not track vertex identity in neighbors; new vertices will not be
                #  duplicates due to visited set tracking
                if vertex != start_vertex:
                    # add found neighbor to start_vertex's neighbors
                    neighbors_at_depth[depth][start_vertex].append(vertex)

                # Check if we've reached the maximum depth; if not look for next level neighbors
                # of found neighbor
                if depth < max_distance:
                    neighbors = set(adjacency_list[vertex]) - visited
                    queue.extend((neighbor, depth + 1) for neighbor in neighbors)
                    visited.update(neighbors)
        connectivity = {}
        for d in range(1, max_distance + 1):
            connectivity[d] = [list(neighbors_at_depth[d][i]) for i in range(vtx_count)]
        return connectivity


class PurePythonDFS(AbstractAdjacencyStrategy):
    # pylint: disable=too-few-public-methods
    """calc adjacencies using DFS in pure python"""

    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Dict[int, List[List[int]]]:
        """calc adjacencies using DFS in pure python"""
        # pylint: disable=too-many-locals
        # values in .values(): indices list -> maximum index + 1 is length of unique vertices
        vtx_count = max(vtx_a for vtxs in hyper_edges_idx.values() for vtx_a in vtxs) + 1
        adjacency_list = _hyper_edges_to_adj_list(vtx_count, hyper_edges_idx)

        neighbors_at_depth = {
            dist: [[] for vertex in range(vtx_count)] for dist in range(max_distance + 1)
        }
        for vertex in range(vtx_count):
            neighbors_at_depth[1][vertex] = adjacency_list[vertex].copy()
            neighbors_at_depth[0][vertex] = [vertex]

        neighbors = [set(adjacency_list[vertex] + [vertex]) for vertex in range(vtx_count)]

        for dist in range(2, max_distance + 1):
            for v in range(vtx_count):
                new_cands = set(
                    x for nn in adjacency_list[v] for x in neighbors_at_depth[dist - 1][nn]
                )
                new = new_cands - neighbors[v]
                neighbors[v].update(new)
                neighbors_at_depth[dist][v] = list(new)
        return neighbors_at_depth
