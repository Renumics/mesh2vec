"""
Plot Neighbor Distances
========================
This example shows ho to plot the distances of nearby elements to a specific element (d<10)
"""

# pylint: disable=pointless-statement

from pathlib import Path
import numpy as np
import pandas as pd
from mesh2vec.mesh2vec_cae import Mesh2VecCae


# %%
# Load Shell from ANSA
# ----------------------
hg = Mesh2VecCae.from_ansa_shell(
    10,
    Path("../../data/hat/Hatprofile.k"),
    json_mesh_file=Path("../../data/hat/cached_hat_key.json"),
)
df = hg.to_dataframe()

# %%
# Find neighbors for specific node and store their distance
# ----------------------------------------------------------
TEST_EID = "1001546"
distances = range(0, 10)
test_neighborhood = [hg.get_nbh(TEST_EID, i) for i in distances]
in_dist_range = np.array(
    [[d if vtx in test_neighborhood[d] else 0 for vtx in hg.vtx_ids()] for d in distances]
)
in_dist_range.shape

# %%
# Check number of neighbors
# --------------------------
np.sum(in_dist_range > 0, axis=1)

# %%
# plot distance for neighbors
# ----------------------------
hg.add_features_from_dataframe(
    pd.DataFrame({"vtx_id": hg.vtx_ids(), "f1": np.sum(in_dist_range, axis=0)})
)
name = hg.aggregate("f1", 0, np.mean)
fig = hg.get_visualization_plotly(str(name))
fig.update_layout(title=name)
fig

# %%
# Access the results
# -------------------
hg.to_dataframe()
