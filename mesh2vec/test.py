from mesh2vec.helpers import BFSAdjacency, BFSNumba, PurePythonBFS, MatMulAdjacency
from mesh2vec.mesh2vec_cae import Mesh2VecCae
from pathlib import Path
import time

start = time.time()
matm = MatMulAdjacency()
a = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
    calc_strategy=matm,
)
print("MATMUL:", time.time() - start)
matm_neighbors = sorted(matm.get_neighbors_exclusive(20, "4700000"))[:20]
print(matm_neighbors)


start = time.time()
bfs = BFSAdjacency()
b = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
    calc_strategy=bfs,
)
print("BFS: ",time.time() - start)
bfs_neighbors = sorted(bfs.get_neighbors_exclusive(20, "4700000"))[:20]
print(bfs_neighbors)

start = time.time()
pbfs = PurePythonBFS()
d = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
    calc_strategy=pbfs,
)
print("PurePythonBFS:", time.time() - start)
pbfs_neighbors = sorted(matm.get_neighbors_exclusive(20, "4700000"))[:20]
print(pbfs_neighbors)


start = time.time()
c = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
    calc_strategy=BFSNumba(),
)
print("NUMBA:", time.time() - start)
numba_neighbors = sorted(matm.get_neighbors_exclusive(20, "4700000"))[:20]
print(numba_neighbors)

trued_neighbors = numba_neighbors==pbfs_neighbors==matm_neighbors==bfs_neighbors
print("IDENTICAL_NEIGHBORS: ", trued_neighbors)