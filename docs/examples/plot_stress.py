"""
Aggregate Y-Stress
====================

"""
# pylint: disable=pointless-statement
from pathlib import Path
import numpy as np
from lasso.dyna import ArrayType
from mesh2vec.mesh2vec_cae import Mesh2VecCae


# %%
# Load Shell from ANSA, simulation results from d3plot
# -----------------------------------------------------
hg = Mesh2VecCae.from_ansa_shell(
    10,
    Path("../../data/hat/Hatprofile.k"),
    json_mesh_file=Path("../../data/hat/cached_hat_key.json"),
)


def mean_over_components_all_layers(v):
    """get mean over all components and all layers"""
    return np.mean(v, axis=-1)


fature_name = hg.add_feature_from_d3plot(
    ArrayType.element_shell_stress,
    Path("../../data/hat/HAT.d3plot"),
    timestep=-1,
    shell_layer=mean_over_components_all_layers,
)


# %%
# Aggregate Feature and plot
# ---------------------------
name = hg.aggregate(fature_name, 1, lambda x: np.mean(np.mean(x)))
fig = hg.get_visualization_plotly(str(name))
fig.update_layout(title=name)
fig


# %%
# Access the results
# -------------------
hg.to_dataframe()
