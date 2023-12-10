"""
measured time and memory usage of mesh2vec 1.0.3
"""

from pathlib import Path
import time
import tracemalloc
import numpy as np
from mesh2vec.mesh2vec_cae import Mesh2VecCae


DIST = 30
PATH = "data/hat/cached_hat_key.json"

PROFILE_MEM = True

# pylint: disable=f-string-without-interpolation
### create neighbors
start = time.time()
if PROFILE_MEM:
    tracemalloc.start()

a = Mesh2VecCae.from_ansa_shell(
    DIST,
    Path("data/hat/Hatprofile.k"),
    json_mesh_file=Path(PATH),
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

a.aggregate_angle_diff(range(DIST), np.mean)

print(f"   aggregate_angle_diff:", time.time() - start)
if PROFILE_MEM:
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()

## tested with p210.key_tmp and mesh2vec 1.0.3
# calc_adjacencies: 41.01193881034851
# add_features_from_ansa: 0.21686553955078125
# 1 aggregate: 27.74431562423706
# 2 aggregate: 55.777445554733276
# aggregate_angle_diff: 427.60096430778503


#    calc_adjacencies: 40.979246377944946
# Current memory usage is 11752.521761MB; Peak was 13671.121043MB
#    add_features_from_ansa: 0.5271070003509521
# Current memory usage is 5.953076MB; Peak was 53.423552MB
#    1 aggregate: 53.40042471885681
# Current memory usage is 6.9341MB; Peak was 103.818146MB
