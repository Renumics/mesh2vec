"""
Quick Start Example
====================
"""

# pylint: disable=pointless-statement

from pathlib import Path
import numpy as np
from mesh2vec.mesh2vec_cae import Mesh2VecCae


# %%
# Load Shell from ANSA
# -----------------------------------------------------
m2v = Mesh2VecCae.from_ansa_shell(
    4,
    Path("../../data/hat/Hatprofile.k"),
    json_mesh_file=Path("../../data/hat/cached_hat_key.json"),
)

# %%
#  Add element features
# ------------------------
m2v.add_features_from_ansa(
    ["aspect", "warpage"],
    Path("../../data/hat/Hatprofile.k"),
    json_mesh_file=Path("../../data/hat/cached_hat_key.json"),
)


# %%
# Aggregate
# ---------------------------
m2v.aggregate("aspect", [0, 2, 3], np.nanmean)


# %%
# Extract Feature Vector
# -----------------------
m2v.to_dataframe()


# %%
# Visualize a single feature
# ---------------------------
m2v.get_visualization_plotly("aspect-nanmean-2")
