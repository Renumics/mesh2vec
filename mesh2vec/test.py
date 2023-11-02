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

# 2002728 n_vtx, index: 1002224
