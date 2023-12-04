"""
Aggregate Y-Stress
====================

"""
# pylint: disable=pointless-statement
from pathlib import Path
import numpy as np
import pandas as pd
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
name = hg.add_feature_from_d3plot(
    ArrayType.element_shell_stress,
    Path("../../data/hat/HAT.d3plot"),
    timestep=-1,
    shell_layer=np.mean,
)


# %%
# Plot Feature locally
# ------------------------
hg.add_features_from_dataframe(
    pd.DataFrame(
        {
            "vtx_id": hg.vtx_ids(),
            "y-stress": [v for v in hg.features()[name]],
        }
    )
)
name = hg.aggregate("y-stress", 0, np.mean)
fig = hg.get_visualization_plotly(str(name))
fig.update_layout(title=name)
fig

# %%
# Aggregate Feature and plot
# ---------------------------
name = hg.aggregate("y-stress", 1, np.mean)
fig = hg.get_visualization_plotly(str(name))
fig.update_layout(title=name)
fig.show()


# %%
# Access the results
# -------------------
hg.to_dataframe()
