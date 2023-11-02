from mesh2vec.mesh2vec_cae import Mesh2VecCae
from pathlib import Path

a = Mesh2VecCae.from_ansa_shell(
    3,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
)

"""# empty adjacency rows to create a csr matrix later on
            row_indices = []
            column_indices = []
            # fill adjacency matrix
            for vertex in graph.vs:
                neighbors = graph.neighborhood(vertices=vertex.index, order=1, mode="all")
                row_indices.extend([vertex.index] * len(neighbors))
                column_indices.extend(neighbors)

            # Create a CSR sparse matrix from row, column, and data lists
            adjacency_matrix = csr_matrix(
                (np.ones(len(row_indices), dtype=int), (row_indices, column_indices)),
                shape=(n_vtx, n_vtx),
            )

            # the desired output type is in array format, not matrix format
            adjacency_matrix_powers_exclusive[depth] = {
                "data": adjacent_matrix.data,
                "indices": adjacent_matrix.indices,
                "indptr": adjacent_matrix.indptr,
                "shape": adjacent_matrix.shape,
            }

            adjacency_matrix_powers[depth] = (
                adjacency_matrix_powers[depth - 1] + adjacency_matrix_powers_exclusive[depth]
            )"""




"""
vertices = [int(vertice) for vertice in hyper_edges_idx.keys()]
        for depth in range(2, max_distance + 1):
            adjacency_matrix = lil_matrix((n_vtx, n_vtx))
            # grab the last cumulative matrix that we calculated as our base; LIL are more efficient than csr matrices
            cumulative_matrix = adjacency_matrix_powers[depth - 1].tolil()
            print("HAHA")
            for vertex in vertices:
                queue = [vertex]
                visited = set()
                current_depth = 0

                while queue and current_depth <= depth:
                    next_level = set()
                    for v in queue:
                        if v not in visited:
                            visited.add(v)
                            adjacency_matrix[vertex, v] = 1
                            cumulative_matrix[vertex, v] = 1
                            neighbors = graph.neighbors(v)
                            next_level.update(neighbors)
                    queue = list(next_level)
                    current_depth += 1

 # convert last cumulative matrix to CSR
        adjacency_matrix_powers[max_distance] = csr_matrix(adjacency_matrix_powers[max_distance]) - csr_matrix(eye(n_vtx, dtype=int))
        print("POWERS BEGINNING ", str(depth), " : ", adjacency_matrix_powers)
        print("POWERS EXCLUSIVE", str(depth), " : ", adjacency_matrix_powers_exclusive)
        print("THIS TOOK: ", time.time() - timer)
"""