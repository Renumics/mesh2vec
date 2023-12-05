"""tests for mesh2vec_hyper_cae"""
from pathlib import Path
from functools import partial

from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from lasso.dyna import ArrayType, D3plot

import pytest
from mesh2vec.mesh_features import CaeShellMesh
from mesh2vec.mesh2vec_cae import (
    Mesh2VecCae,
)


# pylint: disable=protected-access
features_reduced = [
    ArrayType.element_shell_part_indexes,  #: shape (n_shells, 4)
    ArrayType.element_shell_node_indexes,  #: shape (n_shells)
    ArrayType.element_shell_ids,  #: shape (n_shells)
    ArrayType.element_shell_stress,
    #: shape (n_states, n_shells_non_rigid, n_shell_layers, xx_yy_zz_xy_yz_xz)
    ArrayType.element_shell_effective_plastic_strain,
    #: shape (n_states, n_shells_non_rigid, n_shell_layers)
    ArrayType.element_shell_strain,
    #: shape (n_states, n_shells_non_rigid, upper_lower, xx_yy_zz_xy_yz_xz)
    ArrayType.element_shell_is_alive,  #: shape (n_states, n_shells_non_rigid)
]


def _csr_equal(csr_a: csr_matrix, csr_b: csr_matrix) -> bool:
    return all(x == y for x, y in zip(csr_a.todok().items(), csr_b.todok().items()))


def _csr_a_gte_b(csr_a: csr_matrix, csr_b: csr_matrix) -> bool:
    return all(y in csr_a.todok().items() for y in csr_b.todok().items())


def _make_mesh_info(elem_ids: np.ndarray) -> pd.DataFrame:
    element_info = pd.DataFrame({"element_id": elem_ids})
    element_info["part_name"] = "part_name"
    element_info["part_id"] = "part_id"
    element_info["file_path"] = "file_path"
    return element_info


def test_init_tri_unique_ids() -> None:
    """test init with triangles having unique ids"""
    point_coordinates = np.array([[v, v, v] for v in range(6)])
    pnt_ids = np.array(["0", "1", "2", "3x", "4", "5"])
    elem_ids = np.array(["0", "1", "2", "3"])
    elem_node_idxs = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])

    mesh = CaeShellMesh(point_coordinates, pnt_ids, elem_ids, elem_node_idxs)
    m2v = Mesh2VecCae(2, mesh, _make_mesh_info(elem_ids))
    assert ["1", "2"] == m2v.get_nbh("0", 1)


def test_init_tri_overlapping_ids() -> None:
    """test init with triangles having overlapping ids - OPEN TODO"""
    point_coordinates = np.array([[v, v, v] for v in range(6)])
    pnt_ids = np.array(["0", "1", "2x", "3x", "4x", "5"])
    elem_ids = np.array(["0", "1", "0", "1"])
    elem_node_idxs = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])

    mesh = CaeShellMesh(point_coordinates, pnt_ids, elem_ids, elem_node_idxs)
    m2v = Mesh2VecCae(2, mesh, _make_mesh_info(elem_ids))
    assert ["1", "0_2x_3x_4x_4x"] == m2v.get_nbh("0", 1)
    # 2DO: check mesh_point ids


def test_init_quad_overlapping_ids() -> None:
    """test init with quads having overlapping ids"""
    point_coordinates = np.array([[v, v, v] for v in range(6)])
    pnt_ids = np.array(["0", "1", "2", "3x", "4x", "5x"])
    elem_ids = np.array(["0", "1", "0"])
    elem_node_idxs = np.array([[0, 1, 2, 2], [1, 2, 3, 3], [3, 4, 5, 5]])

    mesh = CaeShellMesh(point_coordinates, pnt_ids, elem_ids, elem_node_idxs)
    m2v = Mesh2VecCae(2, mesh, _make_mesh_info(elem_ids))
    assert ["1"] == m2v.get_nbh("0", 1)
    assert ["0", "0_3x_4x_5x_5x"] == m2v.get_nbh("1", 1)


def test_shell_from_ansa() -> None:
    """test ansa shell is loaded"""
    m2v = Mesh2VecCae.from_ansa_shell(
        4,
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path("data/hat/cached_hat_key.json"),
    )
    assert len(m2v.vtx_ids()) == 6400


def test_from_keyfile_shell() -> None:
    """test ansa shell is loaded"""
    m2v = Mesh2VecCae.from_keyfile_shell(
        4,
        Path("data/hat/Hatprofile.k"),
    )

    m2v_ansa = Mesh2VecCae.from_ansa_shell(
        4,
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path("data/hat/cached_hat_key.json"),
    )
    all(sorted(v) == sorted(m2v_ansa._hyper_edges[k]) for k, v in m2v._hyper_edges.items())
    assert set(m2v.vtx_ids()) == set(m2v_ansa.vtx_ids())


def test_shell_from_ansa_partid() -> None:
    """test ansa shell is loaded with and without partid"""
    m2v1 = Mesh2VecCae.from_ansa_shell(
        4,
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path("data/hat/cached_hat_key.json"),
    )

    m2v2 = Mesh2VecCae.from_ansa_shell(
        4,
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path("data/hat/cached_hat_key_partid.json"),
        partid="1",
    )
    assert all(x in m2v1.vtx_ids() for x in m2v2.vtx_ids())


def test_shell_from_d3plot() -> None:
    """test d3plot is loaded"""
    m2v = Mesh2VecCae.from_d3plot_shell(3, Path("data/hat/HAT.d3plot"))
    assert len(m2v.vtx_ids()) > 2000


def test_shell_from_d3plot_partid() -> None:
    """test d3plot is loaded with and without partid"""
    m2v = Mesh2VecCae.from_d3plot_shell(3, Path("data/hat/HAT.d3plot"), partid="1")
    assert len(m2v.vtx_ids()) < 35000


def test_add_features_from_ansa() -> None:
    """test adding features from ansa works"""
    ansafile = Path("data/hat/Hatprofile.k")
    json_mesh_file = Path("data/hat/cached_hat_key.json")

    m2v = Mesh2VecCae.from_ansa_shell(
        4,
        ansafile,
        json_mesh_file=json_mesh_file,
    )
    elements, nodes = Mesh2VecCae._read_ansafile(ansafile, json_mesh_file, verbose=True)
    assert len(elements) == len(m2v._features)
    assert len(nodes) == len(m2v._hyper_edges)

    features = ["aspect", "warpage", "num_border", "is_tria", "midpoint", "normal", "area"]
    m2v.add_features_from_ansa(
        features,
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path("data/hat/cached_hat_key.json"),
    )
    assert all(feature in m2v._features.keys().values for feature in features)


def test_add_features_from_d3plot() -> None:
    """test adding features from d3plot works"""
    m2v = Mesh2VecCae.from_d3plot_shell(3, Path("data/hat/HAT.d3plot"))

    for feature in features_reduced:
        m2v.add_feature_from_d3plot(
            feature,
            Path("data/hat/HAT.d3plot"),
            timestep=1,
            shell_layer=0,
            history_var_index=1,
        )


def test_add_feature_from_d3plot_accumulated() -> None:
    """test adding features from d3plot with shell_layer accumulation works"""
    m2v = Mesh2VecCae.from_d3plot_shell(3, Path("data/hat/HAT.d3plot"))
    d3plot_data = D3plot(Path("data/hat/HAT.d3plot").as_posix())
    axis_0_sum = partial(np.sum, axis=0)
    axis_0_sum.__name__ = "axis0sum"  # type: ignore
    for feature in features_reduced:
        m2v.add_feature_from_d3plot(
            feature, d3plot_data, timestep=1, shell_layer=axis_0_sum, history_var_index=1
        )


def test_save_load() -> None:
    """test saving and loading works"""
    m2v = Mesh2VecCae.from_d3plot_shell(3, Path("data/hat/HAT.d3plot"))
    d3plot_data = D3plot(Path("data/hat/HAT.d3plot").as_posix())
    axis_0_sum = partial(np.sum, axis=0)
    axis_0_sum.__name__ = "axis0sum"  # type: ignore
    for feature in features_reduced:
        m2v.add_feature_from_d3plot(
            feature, d3plot_data, timestep=1, shell_layer=axis_0_sum, history_var_index=1
        )
    with TemporaryDirectory() as tmpdir:
        m2v.save(Path(tmpdir) / "test.h5")
        m2v_loaded = Mesh2VecCae.load(Path(tmpdir) / "test.h5")
        assert m2v_loaded._features.equals(m2v._features)


def test_add_feature_from_d3plot_to_ansa_shell() -> None:
    """test adding features from ansa works"""
    ansafile = Path("data/hat/Hatprofile.k")
    json_mesh_file = Path("data/hat/cached_hat_key.json")

    m2v = Mesh2VecCae.from_ansa_shell(
        4,
        ansafile,
        json_mesh_file=json_mesh_file,
    )

    m2v.add_features_from_ansa(
        ["normal"],
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path("data/hat/cached_hat_key.json"),
    )
    name = m2v.aggregate_angle_diff(2)
    assert isinstance(name, str)
    assert m2v._aggregated_features[name][4] == pytest.approx(0.41053, 0.001)
    # m2v.get_visualization_trimesh(name).show()

    axis_0_sum = partial(np.sum, axis=0)
    axis_0_sum.__name__ = "axis0sum"  # type: ignore

    name_strain = m2v.add_feature_from_d3plot(
        ArrayType.element_shell_strain,
        Path("data/hat/HAT.d3plot"),
        timestep=1,
        shell_layer=axis_0_sum,
    )
    print(m2v._features[name_strain].shape)
    name = m2v.aggregate(name_strain, 1, lambda x: np.mean(np.mean(x)))
    print(m2v._aggregated_features[name].shape)


def test_aggregate_angle_diff() -> None:
    """test aggregating angel difference works"""
    m2v = Mesh2VecCae.from_ansa_shell(
        4, Path("data/hat/Hatprofile.k"), json_mesh_file=Path("data/hat/cached_hat_key.json")
    )
    m2v.add_features_from_ansa(
        ["normal"],
        Path("data/hat/Hatprofile.k"),
        json_mesh_file=Path("data/hat/cached_hat_key.json"),
    )
    name = m2v.aggregate_angle_diff(2)
    assert isinstance(name, str)
    assert m2v._aggregated_features[name][4] == pytest.approx(0.41053, 0.001)
    m2v.get_visualization_trimesh(name)
