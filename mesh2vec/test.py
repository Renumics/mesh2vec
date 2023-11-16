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

print("EXCLUSIVE")
print(matm.get_neighbors_exclusive(3, "4700000"))
"""
start = time.time()
b = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
    calc_strategy=BFSAdjacency(),
)
print("BFS: ",time.time() - start)"""
start = time.time()
pbfs = PurePythonBFS()
d = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
    calc_strategy=pbfs,
)
print("PurePythonBFS:", time.time() - start)

print("EXCLUSIVE")
print(pbfs.get_neighbors_exclusive(3, "4700000"))
print("INCLUSIVE")
print(pbfs.get_neighbors_inclusive(3, "4700000"))

"""start = time.time()
c = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
    calc_strategy=BFSNumba(),
)
print("NUMBA:", time.time() - start)"""


"""ADJACENCY LIST:  [[   0    0]
 [   0    1]
 [   0 2014]
 ...
 [6399 6328]
 [6399 6396]
 [6399 6399]]
DICT 0:  1002224 [0, 2014, 2016, 2202]"""
