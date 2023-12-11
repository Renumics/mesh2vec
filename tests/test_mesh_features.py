"""tests for mesh_features"""

from pathlib import Path

import numpy as np
import pytest

from mesh2vec.mesh_features import (
    area,
    midpoint,
    quads_to_tris_feature_list,
    is_tri,
)
from mesh2vec.mesh2vec_cae import Mesh2VecCae


# pylint: disable=protected-access


def test_area() -> None:
    """test calculated face area is same as in ansa"""

    hg = Mesh2VecCae.from_ansa_shell(
        1,
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path("data/hat/cached_hat_key.json"),
    )

    mesh = hg._mesh
    hg.add_features_from_ansa(
        ["area"],
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path("data/hat/cached_hat_key.json"),
    )

    assert hg._features["area"].values == pytest.approx(
        area(mesh.element_node_idxs, mesh.point_coordinates), abs=1e-1
    )  # high warpage here...


def test_midpoint() -> None:
    """test midpoint calculation with two simple elements (tri/quad)"""
    point_coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    element_node_idxs = np.array([[0, 1, 2, 2], [0, 1, 2, 3]])
    pts = midpoint(element_node_idxs, point_coordinates)
    assert pts == pytest.approx(np.array([[0.33333333, 0.33333333, 0.0], [0.5, 0.5, 0.0]]))


def test_quads_to_tris() -> None:
    """test quads_to_tris works for tris and quads"""
    element_node_idxs = np.array([[0, 1, 2, 2], [0, 1, 2, 3]])
    tri_faces, tri_features = quads_to_tris_feature_list(element_node_idxs, ["red", "green"])
    assert tri_features.tolist() == ["red", "green", "green"]
    assert tri_faces.tolist() == [[0, 1, 2], [0, 1, 2], [0, 2, 3]]


def test_is_tri() -> None:
    """test is_tri works for tris and quads"""
    element_node_idxs = np.array([[0, 1, 2, 2], [0, 1, 2, 3]])
    assert is_tri(element_node_idxs) == [True, False]
