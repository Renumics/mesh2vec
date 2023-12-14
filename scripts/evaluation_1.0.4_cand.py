from mesh2vec.mesh2vec_cae import Mesh2VecCae
from pathlib import Path
import time
import numpy as np
import tracemalloc

DIST = 30
PATH = "data/hat/cached_hat_key.json"
PATH = "/home/markus/Downloads/test_data_internal/p210.key_tmp.json"


res = None
PROFILE_MEM = True
PROFILE_AGG = False

from mesh2vec.helpers import (
    BFSAdjacency,
    BFSNumba,
    PurePythonBFS,
    MatMulAdjacency,
    PurePythonDFS,
    DFSNumba,
    MatMulAdjacencySmart,
)
for strategy in [
    PurePythonDFS,
    MatMulAdjacency,
    MatMulAdjacencySmart,
    BFSAdjacency,
    PurePythonBFS,
    BFSNumba,
    PurePythonDFS,
    DFSNumba,
]:
    print(f"\n########## {strategy}")
    ### create neighbors
    start = time.time()
    if PROFILE_MEM:
        tracemalloc.start()

    a = Mesh2VecCae.from_ansa_shell(
        DIST,
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path(PATH),
        calc_strategy=strategy,
    )
    print(f"   calc_adjacencies:", time.time() - start)
    if PROFILE_MEM:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()

    if PROFILE_MEM:
        tracemalloc.start()
    start = time.time()

    ### load features from ansa
    a.add_features_from_ansa(
        ["warpage", "aspect", "normal", "area"],
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path(PATH),
    )
    print(f"   add_features_from_ansa:", time.time() - start)
    if PROFILE_MEM:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()

    if PROFILE_AGG:
        ### aggregate single feature for 1-30 neighbors
        if PROFILE_MEM:
            tracemalloc.start()
        start = time.time()

        a.aggregate("warpage", range(DIST), np.mean)
        print(f"   1 aggregate:", time.time() - start)
        if PROFILE_MEM:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
            tracemalloc.stop()

        ### aggregate two features for 1-30 neighbors
        if PROFILE_MEM:
            tracemalloc.start()
        start = time.time()

        a.aggregate("warpage", range(DIST), np.mean)
        a.aggregate("aspect", range(DIST), np.mean)
        print(f"   2 aggregate:", time.time() - start)
        if PROFILE_MEM:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
            tracemalloc.stop()

        ### aggregate_angle_diff
        if PROFILE_MEM:
            tracemalloc.start()
        start = time.time()

        a.aggregate_angle_diff(range(DIST), np.mean, skip_arcos=True)

        print(f"   aggregate_angle_diff:", time.time() - start)
        if PROFILE_MEM:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
            tracemalloc.stop()


#    calc_adjacencies: 66.77532052993774
#    add_features_from_ansa: 0.23710179328918457
#    1 aggregate: 2.1715381145477295
#    2 aggregate: 4.20766544342041
#    aggregate_angle_diff: 22.254377841949463


#    calc_adjacencies: 94.87519931793213
# Current memory usage is 537.348588MB; Peak was 13635.465908MB
#    add_features_from_ansa: 0.8933191299438477
# Current memory usage is 5.952401MB; Peak was 53.422866MB
#    1 aggregate: 4.586637496948242
# Current memory usage is 7.003142MB; Peak was 34.636395MB
#    2 aggregate: 9.014816761016846
# Current memory usage is 13.714272MB; Peak was 53.637032MB
#    aggregate_angle_diff: 54.27049207687378
# Current memory usage is 20.440809MB; Peak was 67.720757MB

########## with skip_arcos=True
#    aggregate_angle_diff: 18.00960111618042


########## with different calc_strategies
# ########## <class 'mesh2vec.helpers.PurePythonDFS'>
#    calc_adjacencies: 23.03690767288208
#    add_features_from_ansa: 0.7272367477416992

# ########## <class 'mesh2vec.helpers.MatMulAdjacency'>
#    calc_adjacencies: 64.52890586853027
#    add_features_from_ansa: 0.19918537139892578

# ########## <class 'mesh2vec.helpers.MatMulAdjacencySmart'>
#    calc_adjacencies: 39.02254509925842
#    add_features_from_ansa: 0.292832612991333

# ########## <class 'mesh2vec.helpers.BFSAdjacency'>
#    calc_adjacencies: 30.121832847595215
#    add_features_from_ansa: 0.5960855484008789

# ########## <class 'mesh2vec.helpers.PurePythonBFS'>
#    calc_adjacencies: 50.27418065071106
#    add_features_from_ansa: 0.39345431327819824

# ########## <class 'mesh2vec.helpers.BFSNumba'>
# building adjacency_list
# building neighbors_at_depth
# start numba
# end numba after second 12.097728729248047
#    calc_adjacencies: 82.43846940994263
#    add_features_from_ansa: 0.15757441520690918

# ########## <class 'mesh2vec.helpers.PurePythonDFS'>
#    calc_adjacencies: 24.702327013015747
#    add_features_from_ansa: 0.7912125587463379

# ########## <class 'mesh2vec.helpers.DFSNumba'>
# building adjacency_list
# building neighbors_at_depth
# start numba
# end numba after second 18.648809909820557
#    calc_adjacencies: 88.69992280006409

