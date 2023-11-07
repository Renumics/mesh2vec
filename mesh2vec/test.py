from mesh2vec.helpers import BFSAdjacency
from mesh2vec.mesh2vec_cae import Mesh2VecCae
from pathlib import Path

a = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
)
b = Mesh2VecCae.from_ansa_shell(
    20,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path("data/hat/cached_hat_key.json"),
    calc_strategy=BFSAdjacency(),
)
