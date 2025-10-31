"""
Aggregate Number of Borders Categorical
========================================
"""

# pylint: disable=pointless-statement

from pathlib import Path
import numpy as np
from mesh2vec.mesh2vec_cae import Mesh2VecCae


# %%
# Load Shell and features from ANSA
# ----------------------------------
hg = Mesh2VecCae.from_ansa_shell(
    2,
    Path("../../data/hat/Hatprofile.k"),
    json_mesh_file=Path("../../data/hat/cached_hat_key.json"),
)
hg.add_features_from_ansa(
    ["num_border", "is_tria", "midpoint"],
    ansafile=None,
)


# %%
# Plot Feature locally
# ------------------------
name = hg.aggregate("num_border", 0, np.mean)
fig = hg.get_visualization_plotly(str(name))
fig.update_layout(title=name)
fig


# %%
# Aggregate Feature and plot
# ---------------------------
names = hg.aggregate_categorical("num_border", 1)
NAME = "num_border-cat-1-1"
fig = hg.get_visualization_plotly(str(name))
fig.update_layout(title=NAME)
fig

# %%
# Access the results
# -------------------
hg.to_dataframe()
