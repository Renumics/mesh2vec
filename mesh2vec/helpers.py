"""helper functions"""
from typing import OrderedDict, List, Tuple, Dict
from numba import jit
import numpy as np
from collections import deque
from scipy.sparse import csr_array, coo_array, eye
import igraph as ip
import itertools
from abc import ABC, abstractmethod

"""
Use strategy to implement multiple adjacency finding algorithms
"""

@jit(parallel=False, fastmath=True, forceobj=True)
def numba_compute(adj_list, depth, vtx_count):
    # each vertex has a list of lists where the index of the list corresponds to the depth of its contained neighbors
    # e.g.: neighbor 300 at depth 2, neighbor 400 at depth 1 with max_distance 3 -> [[], [400], [300], []]
    neighbors_at_depth = {
        vertex: [np.array([], dtype=np.int64) for _ in range(depth + 1)] for vertex in range(vtx_count)
    }

    # for each vertex, do bfs separately
    for start_vertex in range(vtx_count):
        # Initialize the queue for BFS: (vertex, depth); numba does not support dequeue
        queue = [(start_vertex, 0)]

        # Use a set to keep track of visited vertices
        visited = [start_vertex]

        # queue not empty
        while queue:
            # depth is saved in queue -> no tracking in loop
            vertex, depth = queue.pop()

            # do not track vertex identity in neighbors; new vertices will not be duplicates due
            # to visited set tracking
            if vertex != start_vertex:
                # add found neighbor to start_vertex's neighbors
                neighbors_at_depth[start_vertex][depth] = np.append(neighbors_at_depth[start_vertex][depth], vertex)

            # Check if we've reached the maximum depth; if not look for next level neighbors
            # of found neighbor
            if depth < depth:
                # remove duplicates manually, as np.setdiff1d and - with lists is not supported
                neighbors = np.unique(np.array(adj_list[vertex]))
                removed_neighbors = np.array([neighbor for neighbor in neighbors if neighbor not in visited], dtype=np.int64)
                        
                for neighbor in removed_neighbors:
                    queue.append((neighbor, depth + 1))
                visited.extend(removed_neighbors)
        

    return neighbors_at_depth


class AbstractAdjacencyStrategy(ABC):
    def __init__(self):
        self.adjacency_matrix_powers = None
        self.adjacency_matrix_powers_exclusive = None
        self.idx_conversion = None
        self.id_conversion = None

    def set_matrices(self, matrix_powers, matrix_powers_exclusive):
        self.adjacency_matrix_powers = matrix_powers
        self.adjacency_matrix_powers_exclusive = matrix_powers_exclusive

    def set_idx_conversion(self, translation: OrderedDict):
        # key -> vtx_id, value -> index
        self.idx_conversion = translation

    def set_id_conversion(self, translation: OrderedDict):
        # key -> vtx_idx, value -> vtx_id
        self.id_conversion = translation

    @abstractmethod
    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Tuple[Dict[int, csr_array], Dict[int, csr_array]]:
        pass

    @abstractmethod
    def get_neighbors_exclusive(
        self, distance: int, vtx_id: int, transformation: bool = True
    ) -> List[int]:
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

    def get_neighbors_exclusive(
        self, distance: int, vtx_id: int, transformation: bool = True
    ) -> List[int]:
        row_index = self.idx_conversion[vtx_id]
        # get the indices;
        selected_row = self.adjacency_matrix_powers_exclusive[distance][[row_index], :]
        # nonzero returns a tuple of ([row], [columns]); since we already selected the correct row:
        # look at columns (= vtx_indices)
        nonzero_indices = selected_row.nonzero()[1]
        if transformation:
            # convert to vtx_ids
            nonzero_ids = [self.id_conversion[index] for index in nonzero_indices]
            return nonzero_ids
        return nonzero_indices

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
        adjacency_list_np: np.ndarray = np.unique(np.array(adjacency_list), axis=0)

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

    def get_neighbors_exclusive(
        self, distance: int, vtx_id: int, transformation: bool = True
    ) -> List[int]:
        if not transformation:
            return self.adjacency_matrix_powers_exclusive[distance][self.idx_conversion[vtx_id]]

        matrix = self.adjacency_matrix_powers_exclusive[distance][self.idx_conversion[vtx_id]]
        return [self.id_conversion[vtx] for vtx in matrix]

    def __str__(self) -> str:
        return "BFSAdjacency"


class BFSNumba(AbstractAdjacencyStrategy):
    def __init__(self):
        super().__init__()

    def get_neighbors_exclusive(self, distance: int, vtx_id: int) -> List[int]:
        # TODO CONVERSION
        return self.adjacency_matrix_powers_exclusive[distance][self.idx_conversion[vtx_id]]

    def __str__(self) -> str:
        return "BFSNumba"

    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Tuple[Dict[int, csr_array], Dict[int, csr_array]]:
        # values in .values(): indices list -> maximum index + 1 is length of unique vertices
        vtx_count = max([vtx_a for vtxs in hyper_edges_idx.values() for vtx_a in vtxs]) + 1
        adjacency_list = [[] for _ in range(vtx_count)]
        for vtxs in hyper_edges_idx.values():
            for vtx_a in vtxs:
                for vtx_b in vtxs:
                    adjacency_list[vtx_a].append(vtx_b)
                    adjacency_list[vtx_b].append(vtx_a)
        for i in range(len(adjacency_list)):
            adjacency_list[i] = list(set(adjacency_list[i]))

        neighbors_at_depth = numba_compute(adjacency_list, max_distance, vtx_count)

        self.adjacency_matrix_powers_exclusive = neighbors_at_depth
        self.set_matrices(neighbors_at_depth, neighbors_at_depth)

        # constructing an inclusive neighbor matrix/ dictionary would be expensive; use get_neighbors_inclusive
        # to get inclusive neighbors for specific vertex. This avoids computational overhead
        return neighbors_at_depth, neighbors_at_depth


class PurePythonBFS(AbstractAdjacencyStrategy):
    def __init__(self):
        super().__init__()

    def __str__(self) -> str:
        return "PurePythonBFS"

    def get_neighbors_exclusive(
        self, distance: int, vtx_id: int, transformation: bool = True
    ) -> List[int]:
        row_index = self.idx_conversion[vtx_id]
        idx_list = self.adjacency_matrix_powers_exclusive[row_index][distance]
        if transformation:
            return [self.id_conversion[idx] for idx in idx_list]
        return idx_list

    def get_neighbors_inclusive(
        self, distance: int, vtx_id: int, transformation: bool = True
    ) -> List[int]:
        # join all sublists of neighbors into one
        row_index = self.idx_conversion[vtx_id]
        idx_list = list(
            itertools.chain(*self.adjacency_matrix_powers_exclusive[row_index][:distance])
        )
        if transformation:
            return [self.id_conversion[index] for index in idx_list]
        return idx_list

    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Tuple[List[List[int]], List[List[int]]]:
        # values in .values(): indices list -> maximum index + 1 is length of unique vertices
        vtx_count = max([vtx_a for vtxs in hyper_edges_idx.values() for vtx_a in vtxs]) + 1
        adjacency_list = [[] for _ in range(vtx_count)]
        for vtxs in hyper_edges_idx.values():
            for vtx_a in vtxs:
                for vtx_b in vtxs:
                    adjacency_list[vtx_a].append(vtx_b)
                    adjacency_list[vtx_b].append(vtx_a)
        for i in range(len(adjacency_list)):
            adjacency_list[i] = list(set(adjacency_list[i]))

        # each vertex has a list of lists where the index of the list corresponds to the depth of its contained neighbors
        # e.g.: neighbor 300 at depth 2, neighbor 400 at depth 1 with max_distance 3 -> [[], [400], [300], []]
        neighbors_at_depth = {
            vertex: [[] for _ in range(max_distance + 1)] for vertex in range(vtx_count)
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

                # do not track vertex identity in neighbors; new vertices will not be duplicates due
                # to visited set tracking
                if vertex != start_vertex:
                    # add found neighbor to start_vertex's neighbors
                    neighbors_at_depth[start_vertex][depth].append(vertex)

                # Check if we've reached the maximum depth; if not look for next level neighbors
                # of found neighbor
                if depth < max_distance:
                    neighbors = set(adjacency_list[vertex]) - visited
                    queue.extend((neighbor, depth + 1) for neighbor in neighbors)
                    visited.update(neighbors)

        self.adjacency_matrix_powers_exclusive = neighbors_at_depth
        self.set_matrices(neighbors_at_depth, neighbors_at_depth)

        # constructing an inclusive neighbor matrix/ dictionary would be expensive; use get_neighbors_inclusive
        # to get inclusive neighbors for specific vertex. This avoids computational overhead
        return neighbors_at_depth, neighbors_at_depth
