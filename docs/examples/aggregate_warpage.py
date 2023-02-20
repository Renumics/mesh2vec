"""
Aggregate Warpage
===================
"""
# pylint: disable=pointless-statement
from pathlib import Path
import numpy as np
from mesh2vec.mesh2vec_cae import Mesh2VecCae


# %%
# Load Shell from ANSA, simulation results from d3plot
# -----------------------------------------------------
hg = Mesh2VecCae.from_ansa_shell(
    3,
    Path("../../data/hat/Hatprofile.k"),
    json_mesh_file=Path("../../data/hat/cached_hat_key.json"),
)

# %%
# Plot Feature locally
# ------------------------
hg.add_features_from_ansa(
    ["warpage"],
    Path("../../data/hat/Hatprofile.k"),
    json_mesh_file=Path("../../data/hat/cached_hat_key.json"),
)
name = hg.aggregate("warpage", 0, np.nanmean)
fig = hg.get_visualization_plotly(str(name))
fig.update_layout(title=name[0])
fig

# %%
# Aggregate Feature and plot
# ---------------------------
name = hg.aggregate("warpage", 3, np.nanmean)
fig = hg.get_visualization_plotly(str(name))
fig.update_layout(title=name)
fig

# %%
# Access the results
# -------------------
hg.to_dataframe()
