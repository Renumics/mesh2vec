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


"""start = time.time()
c = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
    calc_strategy=BFSNumba(),
)
print("NUMBA:", time.time() - start)"""
