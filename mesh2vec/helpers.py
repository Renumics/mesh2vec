"""helper functions"""
from typing import OrderedDict, List, Dict
from collections import deque
from abc import ABC, abstractmethod
import time

from numba import jit
import numpy as np
from scipy.sparse import csr_array, coo_array, eye
import igraph as ip
import numba
import numba.typed

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


class MatMulAdjacencySmart(AbstractAdjacencyStrategy):
    """calc adjacencies using matrix multiplication with low memory footprint"""

    # pylint: disable=too-few-public-methods
    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Dict[int, List[List[int]]]:
        """calc adjacencies using matrix multiplication with low memory footprint"""
        adjacency_list_np = _hyper_edges_to_adj_np(hyper_edges_idx)

        # adjacency matrix
        n_vtx = max(max(v) for v in hyper_edges_idx.values()) + 1
        data = np.array([1] * len(adjacency_list_np))
        adjacency_matrix = coo_array(
            (data, (adjacency_list_np[:, 1], adjacency_list_np[:, 0])), shape=(n_vtx, n_vtx)
        ).tocsr()

        # neighbors
        adjacency_matrix_powers_exclusive = {
            1: adjacency_matrix - csr_array(eye(n_vtx, dtype=int)),
            0: csr_array(eye(n_vtx, dtype=int)),
        }

        all_neighbors = (
            adjacency_matrix_powers_exclusive[1] + adjacency_matrix_powers_exclusive[0]
        )

        for i in range(2, max_distance + 1):
            new = adjacency_matrix_powers_exclusive[i - 1] @ adjacency_matrix
            new.data = np.array([1] * new.nnz)  # make all entries 1

            exclusive = new > all_neighbors
            all_neighbors = all_neighbors + new

            adjacency_matrix_powers_exclusive[i] = exclusive

        neighborhoods = {}
        for dist in range(1, max_distance + 1):
            neighborhoods[dist] = [
                adjacency_matrix_powers_exclusive[dist][[i], :].indices for i in range(n_vtx)
            ]
        return neighborhoods


class MatMulAdjacency(AbstractAdjacencyStrategy):
    # pylint: disable=too-few-public-methods
    """calc adjacencies using matrix multiplication with low memory footprint"""

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


class BFSAdjacency(AbstractAdjacencyStrategy):
    """calc adjacencies using BFS from igraph"""

    # pylint: disable=too-few-public-methods
    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Dict[int, List[List[int]]]:
        """calc adjacencies using BFS from igraph"""
        adjacency_list_np: np.ndarray = _hyper_edges_to_adj_np(hyper_edges_idx)

        graph = ip.Graph(adjacency_list_np)

        ####################################################
        # General idea: BFS for each vertex for each depth #
        ####################################################
        neighborhoods = {}
        n_vtx = max(max(v) for v in hyper_edges_idx.values()) + 1

        for depth in range(1, max_distance + 1):

            exclusive_neighborhood = graph.neighborhood(vertices=None, mindist=depth, order=depth)
            neighborhoods[depth] = [
                list(set(exclusive_neighborhood[i]) - {i}) for i in range(n_vtx)
            ]

        return neighborhoods


class BFSNumba(AbstractAdjacencyStrategy):
    """calc adjacencies using BFS in numba"""

    # pylint: disable=too-few-public-methods
    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Dict[int, List[List[int]]]:
        """calc adjacencies using BFS in numba"""
        vtx_count = max(vtx_a for vtxs in hyper_edges_idx.values() for vtx_a in vtxs) + 1

        adjacency_list = _hyper_edges_to_adj_list(vtx_count, hyper_edges_idx)

        print("building adjacency_list")
        adjacency_list_numba = numba.typed.List()
        for _ in range(vtx_count):
            new = numba.typed.List()
            new.append(1)
            new.pop()
            adjacency_list_numba.append(new)

        for i, neighbors in enumerate(adjacency_list):
            for neighbor in neighbors:
                adjacency_list_numba[i].append(neighbor)

        # neighbors_at_depth = List(List(  List([-1]) for _ in range(max_distance + 1))
        #   for vertex in range(vtx_count))
        print("building neighbors_at_depth")
        neighbors_at_depth = numba.typed.List()
        for i in range(vtx_count):
            new = numba.typed.List()
            for _ in range(max_distance + 1):
                new_new = numba.typed.List()
                new_new.append(1)
                new_new.pop()
                new.append(new_new)
            neighbors_at_depth.append(new)

        print("start numba")
        start = time.time()

        neighbors_at_depth = _numba_compute_bfs(
            adjacency_list_numba, max_distance, vtx_count, neighbors_at_depth
        )
        print(f"end numba after second {time.time() - start}")

        connectivity = {}
        for d in range(1, max_distance + 1):
            connectivity[d] = [list(neighbors_at_depth[i][d]) for i in range(vtx_count)]
        return connectivity


@jit(nopython=True, parallel=False, fastmath=True)
def _numba_compute_bfs(adj_list, max_depth, vtx_count, neighbors_at_depth):
    """
    compute bfs in numba
    Args:
        adj_list: adjacency list as numba typed list of lists
        max_depth: maximum depth to search
        vtx_count: number of vertices
        neighbors_at_depth: empty list of lists of lists (vtx, depth, neighbors)

    """
    for start_vertex in range(vtx_count):
        # Initialize the queue for BFS: (vertex, depth); numba does not support dequeue
        queue = [(start_vertex, 0)]

        # Use a set to keep track of visited vertices
        # visited = {start_vertex}
        visited_array = np.array([False] * vtx_count)

        # queue not empty
        while queue:
            # depth is saved in queue -> no tracking in loop
            vertex, depth = queue.pop(0)

            # do not track vertex identity in neighbors; new vertices will not be duplicates due
            # to visited set tracking
            if vertex != start_vertex:
                # add found neighbor to start_vertex's neighbors
                neighbors_at_depth[start_vertex][depth].append(vertex)

            # Check if we've reached the maximum depth; if not look for next level neighbors
            # of found neighbor
            if depth < max_depth:
                neighbors = [x for x in adj_list[vertex] if not visited_array[x]]
                for neighbor in neighbors:
                    queue.append((neighbor, depth + 1))
                    visited_array[neighbor] = True

    return neighbors_at_depth


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


class DFSNumba(AbstractAdjacencyStrategy):
    # pylint: disable=too-few-public-methods,too-many-locals
    """calc adjacencies using DFS in numba"""

    def calc_adjacencies(
        self, hyper_edges_idx: OrderedDict[str, List[int]], max_distance: int
    ) -> Dict[int, List[List[int]]]:
        """calc adjacencies using DFS in numba"""
        # values in .values(): indices list -> maximum index + 1 is length of unique vertices
        vtx_count = max(vtx_a for vtxs in hyper_edges_idx.values() for vtx_a in vtxs) + 1
        adjacency_list = _hyper_edges_to_adj_list(vtx_count, hyper_edges_idx)

        print("building adjacency_list")
        adjacency_list_numba = numba.typed.List()
        for _ in range(vtx_count):
            new = numba.typed.List()
            new.append(1)
            new.pop()
            adjacency_list_numba.append(new)

        for i, neighbors in enumerate(adjacency_list):
            for neighbor in neighbors:
                adjacency_list_numba[i].append(neighbor)

        print("building neighbors_at_depth")
        neighbors_at_depth = numba.typed.List()
        for i in range(vtx_count):
            new = numba.typed.List()

            vtx_self = numba.typed.List()
            vtx_self.append(i)
            new.append(vtx_self)  # 0

            adjs = numba.typed.List()
            for nn in adjacency_list_numba[i]:
                adjs.append(nn)

            new.append(adjs)  # 1

            for _ in range(max_distance + 1):
                new_new = numba.typed.List()
                new_new.append(1)
                new_new.pop()
                new.append(new_new)
            neighbors_at_depth.append(new)

        print("start numba")
        start = time.time()

        neighbors_at_depth = _numba_compute_dfs(
            adjacency_list_numba, max_distance, vtx_count, neighbors_at_depth
        )
        print(f"end numba after second {time.time() - start}")

        connectivity = {}
        for d in range(1, max_distance + 1):
            connectivity[d] = [list(neighbors_at_depth[i][d]) for i in range(vtx_count)]
        return connectivity


@jit(nopython=True, parallel=False, fastmath=True)
def _numba_compute_dfs(adj_list, max_distance, vtx_count, neighbors_at_depth):
    """
    compute dfs in numba
    Args:
        adj_list: adjacency list as numba typed list of lists
        max_distance: maximum depth to search
        vtx_count: number of vertices
        neighbors_at_depth: empty list of lists of lists (vtx, depth, neighbors)

    """
    neighbors = numba.typed.List()
    for vertex in range(vtx_count):
        neighbors_current = set(adj_list[vertex])
        neighbors_current.add(vertex)
        neighbors.append(neighbors_current)

    for dist in range(2, max_distance + 1):
        for v in range(vtx_count):
            hood = set()
            for nn in adj_list[v]:
                hood.update(neighbors_at_depth[nn][dist - 1])
            hood = hood - neighbors[v]
            neighbors[v].update(hood)
            neighbors_at_depth[v][dist] = numba.typed.List(hood)
    return neighbors_at_depth
