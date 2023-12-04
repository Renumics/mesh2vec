"""tests for mesh2vec_hyper"""
from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csr import csr_matrix

from mesh2vec.mesh2vec_base import Mesh2VecBase
from mesh2vec.mesh2vec_exceptions import (
    InvalidDistanceArgumentException,
    InvalidHyperEdgesArgument,
    InvalidVtxIdsArgument,
    InvalidVtxIdArgument,
    FeatureDoesNotExistException,
)

from mesh2vec.helpers import (
    MatMulAdjacency,
    BFSAdjacency,
    BFSNumba,
    PurePythonBFS,
    PurePythonDFS,
    DFSNumba,
    MatMulAdjacencySmart,
)


strategies = [
    MatMulAdjacency,
    BFSAdjacency,
    BFSNumba,
    PurePythonBFS,
    PurePythonDFS,
    DFSNumba,
    MatMulAdjacencySmart,
]

# pylint: disable=protected-access
def _csr_equal(csr_a: csr_matrix, csr_b: csr_matrix) -> bool:
    return all(x == y for x, y in zip(csr_a.todok().items(), csr_b.todok().items()))


def _csr_a_gte_b(csr_a: csr_matrix, csr_b: csr_matrix) -> bool:
    return all(y in csr_a.todok().items() for y in csr_b.todok().items())


def test_init() -> None:
    """test init"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"]}
    hg = Mesh2VecBase(3, edges)
    assert hg._distance == 3
    assert "y" in hg._vtx_ids_to_idx.keys()
    assert set(hg._neighborhoods[1][0]) == {1, 2}  # a->b,c


def test_invalid_hyperedges() -> None:
    """test that InvalidDistanceArgument is raised"""

    with pytest.raises(InvalidHyperEdgesArgument):
        edges_0 = {"first": ["a", 0.1, "c"], "second": ["x", "y"], "third": ["x", "a"]}
        _ = Mesh2VecBase(3, edges_0)  # type: ignore

    with pytest.raises(InvalidHyperEdgesArgument):
        edges_1 = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", 123]}
        _ = Mesh2VecBase(3, edges_1)  # type: ignore

    with pytest.raises(InvalidHyperEdgesArgument):
        edges_2 = [[1, 2, 3], [2, 3, 4]]
        _ = Mesh2VecBase(3, edges_2)  # type: ignore


def test_invalid_vtx_ids_argument() -> None:
    """test that InvalidVtxIdsArgument is raised"""
    with pytest.raises(InvalidVtxIdsArgument):
        edges_0 = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
        _ = Mesh2VecBase(3, edges_0, vtx_ids=[1, "b", "x", "y"])  # type: ignore

    with pytest.raises(InvalidVtxIdsArgument):
        edges_1 = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
        _ = Mesh2VecBase(3, edges_1, vtx_ids=["a", "b", "x", "y"])


def test_from_txt_file() -> None:
    """test import of txt file creates a consistent hypergraph"""
    hg = Mesh2VecBase.from_file(Path("data/hyper_02.txt"), 3)
    assert hg._distance == 3
    assert "vtx02" in hg._vtx_ids_to_idx.keys()


def test_from_csv_file() -> None:
    """test import of csv file creates a consistent hypergraph"""
    hg1 = Mesh2VecBase.from_file(Path("data/hyper_02.txt"), 3)

    # write pair-wise connectivity to csv
    connections = []
    connectivity_list = [x for x in hg1._neighborhoods[1]]
    for vtx_a, vtx_a_neigbors in enumerate(connectivity_list):
        for vtx_b in vtx_a_neigbors:
            connections.append([hg1._vtx_idx_to_ids[vtx_a], hg1._vtx_idx_to_ids[vtx_b]])
    df = pd.DataFrame(connections, columns=["vtx_a", "vtx_b"])
    df.to_csv("data/tmp_hyper_02.csv", header=False, index=False)

    hg2 = Mesh2VecBase.from_file(Path("data/tmp_hyper_02.csv"), 3)
    assert hg2._distance == 3

    # adj_mat must be equal
    for i in range(1, 3 + 1):
        for j in range(13):
            assert all(hg2._neighborhoods[i][j] == hg1._neighborhoods[i][j])

    # hyper edges of hg1 that are not fully contained in larger other hyper edge form a clique
    # and should be contained in hyper edges of hg2
    sorted_hg2_hyper_edges = [sorted(hyper_edge) for hyper_edge in hg2._hyper_edges.values()]
    for hyper_edge in hg1._hyper_edges.values():
        if not any(
            np.all(
                list(
                    [v in hyper_edge_2 for v in hyper_edge]
                    for hyper_edge_2 in hg1._hyper_edges.values()
                    if not hyper_edge_2 == hyper_edge
                ),
                axis=1,
            )
        ):
            assert sorted(hyper_edge) in sorted_hg2_hyper_edges


@pytest.mark.parametrize("strategy", strategies)
def test_get_nbh(strategy) -> None:
    """test get_nbh"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
    hg = Mesh2VecBase(3, edges, calc_strategy=strategy)
    assert {"b"} == set(hg.get_nbh("b", 0))
    assert {"a", "c"} == set(hg.get_nbh("b", 1))
    assert {"x"} == set(hg.get_nbh("y", 1))

    assert {"x"} == set(hg.get_nbh("b", 2))
    assert {"a"} == set(hg.get_nbh("y", 2))

    assert {"y"} == set(hg.get_nbh("b", 3))
    assert {"b", "c"} == set(hg.get_nbh("y", 3))


def test_invalid_distance_exception() -> None:
    """test that InvalidDistanceArgument is raised"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}

    with pytest.raises(InvalidDistanceArgumentException):
        _ = Mesh2VecBase(-3, edges)
    with pytest.raises(InvalidDistanceArgumentException):
        hg = Mesh2VecBase(3, edges)
        hg.get_nbh("b", 4)


def test_invalidvtx_id_exception() -> None:
    """test that InvalidDistanceArgument is raised"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}

    with pytest.raises(InvalidVtxIdArgument):
        hg = Mesh2VecBase(3, edges)
        hg.get_nbh(42, 3)  # type: ignore

    with pytest.raises(InvalidVtxIdArgument):
        hg = Mesh2VecBase(3, edges)
        hg.get_nbh("42", 3)


def test_aggregate_categorical_disjunctive() -> None:
    """test categorical aggregation with simple with disjunctive feature values"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"]}
    hg = Mesh2VecBase(3, edges)
    df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": [2, 4, 8, 16, 32]})
    hg.add_features_from_dataframe(df1)
    _ = hg.aggregate_categorical("f1", [1, 3])

    assert hg._aggregated_features["f1-cat-2-1"].tolist() == [0, 1, 1, 0, 0]
    assert hg._aggregated_features["f1-cat-4-1"].tolist() == [1, 0, 1, 0, 0]
    assert hg._aggregated_features["f1-cat-8-1"].tolist() == [1, 1, 0, 0, 0]
    assert hg._aggregated_features["f1-cat-16-1"].tolist() == [0, 0, 0, 0, 1]
    assert hg._aggregated_features["f1-cat-32-1"].tolist() == [0, 0, 0, 1, 0]


@pytest.mark.parametrize("feature_values", [[2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"]])
def test_aggregate_categorical_disjunctive_complex(feature_values: List[Any]) -> None:
    """test categorical aggregation with complex graph but disjunctive values"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
    hg = Mesh2VecBase(3, edges)
    df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": feature_values})
    hg.add_features_from_dataframe(df1)

    _ = hg.aggregate_categorical("f1", 1)
    assert hg._aggregated_features["f1-cat-2-1"].tolist() == [0, 1, 1, 1, 0]
    assert hg._aggregated_features["f1-cat-4-1"].tolist() == [1, 0, 1, 0, 0]
    assert hg._aggregated_features["f1-cat-8-1"].tolist() == [1, 1, 0, 0, 0]
    assert hg._aggregated_features["f1-cat-16-1"].tolist() == [1, 0, 0, 0, 1]
    assert hg._aggregated_features["f1-cat-32-1"].tolist() == [0, 0, 0, 1, 0]

    _ = hg.aggregate_categorical("f1", 2)
    assert hg._aggregated_features["f1-cat-2-2"].tolist() == [0, 0, 0, 0, 1]
    assert hg._aggregated_features["f1-cat-4-2"].tolist() == [0, 0, 0, 1, 0]
    assert hg._aggregated_features["f1-cat-8-2"].tolist() == [0, 0, 0, 1, 0]
    assert hg._aggregated_features["f1-cat-16-2"].tolist() == [0, 1, 1, 0, 0]
    assert hg._aggregated_features["f1-cat-32-2"].tolist() == [1, 0, 0, 0, 0]

    _ = hg.aggregate_categorical("f1", 3)
    assert hg._aggregated_features["f1-cat-2-3"].tolist() == [0, 0, 0, 0, 0]
    assert hg._aggregated_features["f1-cat-4-3"].tolist() == [0, 0, 0, 0, 1]
    assert hg._aggregated_features["f1-cat-8-3"].tolist() == [0, 0, 0, 0, 1]
    assert hg._aggregated_features["f1-cat-16-3"].tolist() == [0, 0, 0, 0, 0]
    assert hg._aggregated_features["f1-cat-32-3"].tolist() == [0, 1, 1, 0, 0]


@pytest.mark.parametrize("feature_values", [[2, 4, 8, 2, 4], ["2", "4", "8", "2", "4"]])
def test_aggregate_categorical_ones(feature_values: List[Any]) -> None:
    """test categorical aggregation with complex graph but only ones in sums"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
    hg = Mesh2VecBase(3, edges)
    df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": feature_values})
    hg.add_features_from_dataframe(df1)

    _ = hg.aggregate_categorical("f1", 1)
    assert hg._aggregated_features["f1-cat-2-1"].tolist() == [1, 1, 1, 1, 1]
    assert hg._aggregated_features["f1-cat-4-1"].tolist() == [1, 0, 1, 1, 0]
    assert hg._aggregated_features["f1-cat-8-1"].tolist() == [1, 1, 0, 0, 0]

    _ = hg.aggregate_categorical("f1", 2)
    assert hg._aggregated_features["f1-cat-2-2"].tolist() == [0, 1, 1, 0, 1]
    assert hg._aggregated_features["f1-cat-4-2"].tolist() == [1, 0, 0, 1, 0]
    assert hg._aggregated_features["f1-cat-8-2"].tolist() == [0, 0, 0, 1, 0]

    _ = hg.aggregate_categorical("f1", 3)
    assert hg._aggregated_features["f1-cat-2-3"].tolist() == [0, 0, 0, 0, 0]
    assert hg._aggregated_features["f1-cat-4-3"].tolist() == [0, 1, 1, 0, 1]
    assert hg._aggregated_features["f1-cat-8-3"].tolist() == [0, 0, 0, 0, 1]


@pytest.mark.parametrize("feature_values", [[2, 4, 2, 2, 4], ["2", "4", "2", "2", "4"]])
def test_aggregate_categorical_mores(feature_values: List[Any]) -> None:
    """test categorical aggregation with complex graph and sums > 1"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
    hg = Mesh2VecBase(3, edges)
    df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": feature_values})
    hg.add_features_from_dataframe(df1)

    _ = hg.aggregate_categorical("f1", 0)
    assert hg._aggregated_features["f1-cat-2-0"].tolist() == [1, 0, 1, 1, 0]
    assert hg._aggregated_features["f1-cat-4-0"].tolist() == [0, 1, 0, 0, 1]

    _ = hg.aggregate_categorical("f1", 1)
    assert hg._aggregated_features["f1-cat-2-1"].tolist() == [2, 2, 1, 1, 1]
    assert hg._aggregated_features["f1-cat-4-1"].tolist() == [1, 0, 1, 1, 0]

    _ = hg.aggregate_categorical("f1", 2)
    assert hg._aggregated_features["f1-cat-2-2"].tolist() == [0, 1, 1, 1, 1]
    assert hg._aggregated_features["f1-cat-4-2"].tolist() == [1, 0, 0, 1, 0]

    _ = hg.aggregate_categorical("f1", 3)
    assert hg._aggregated_features["f1-cat-2-3"].tolist() == [0, 0, 0, 0, 1]
    assert hg._aggregated_features["f1-cat-4-3"].tolist() == [0, 1, 1, 0, 1]


def test_aggregates_raise_feature_not_available() -> None:
    """test that FeatureDoesNotExistException is raised"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"]}
    hg = Mesh2VecBase(3, edges)
    df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": [2, 4, 8, 16, 32]})
    hg.add_features_from_dataframe(df1)
    with pytest.raises(FeatureDoesNotExistException, match=r"\['f1'\]"):
        _ = hg.aggregate("f_not_exist", 1, np.mean)
    with pytest.raises(FeatureDoesNotExistException, match=r"\['f1'\]"):
        _ = hg.aggregate_categorical("f_not_exist", 1)


@pytest.mark.parametrize("strategy", strategies)
def test_aggregate_simple(strategy) -> None:
    """test aggregation with simple graph"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"]}
    hg = Mesh2VecBase(3, edges, calc_strategy=strategy)
    df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": [2, 4, 8, 16, 32]})
    hg.add_features_from_dataframe(df1)
    name = hg.aggregate("f1", 1, np.mean)
    assert hg._aggregated_features[name].iloc[0] == np.mean([4, 8])
    assert hg._aggregated_features[name].iloc[1] == np.mean([2, 8])
    assert hg._aggregated_features[name].iloc[2] == np.mean([2, 4])
    assert hg._aggregated_features[name].iloc[3] == np.mean([32])
    assert hg._aggregated_features[name].iloc[4] == np.mean([16])


def test_aggregate_simple_two_dists() -> None:
    """test aggregation with simple graph"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"]}
    hg = Mesh2VecBase(3, edges)
    df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": [2, 4, 8, 16, 32]})
    hg.add_features_from_dataframe(df1)
    names = hg.aggregate("f1", [1, 2], np.mean)
    assert hg._aggregated_features[names[0]].iloc[0] == np.mean([4, 8])
    assert hg._aggregated_features[names[0]].iloc[1] == np.mean([2, 8])
    assert hg._aggregated_features[names[0]].iloc[2] == np.mean([2, 4])
    assert hg._aggregated_features[names[0]].iloc[3] == np.mean([32])
    assert hg._aggregated_features[names[0]].iloc[4] == np.mean([16])
    assert hg._aggregated_features[names[1]].iloc[0] == 0


def test_aggregate_complex() -> None:
    """test aggregation with complex graph"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
    hg = Mesh2VecBase(3, edges)
    df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": [2, 4, 8, 16, 32]})
    hg.add_features_from_dataframe(df1)

    name = hg.aggregate("f1", 0, np.mean)
    assert hg._aggregated_features[name].iloc[0] == 2
    assert hg._aggregated_features[name].iloc[1] == 4
    assert hg._aggregated_features[name].iloc[2] == 8
    assert hg._aggregated_features[name].iloc[3] == 16
    assert hg._aggregated_features[name].iloc[4] == 32

    name = hg.aggregate("f1", 1, np.mean)
    assert hg._aggregated_features[name].iloc[0] == np.mean([4, 8, 16])
    assert hg._aggregated_features[name].iloc[1] == np.mean([2, 8])
    assert hg._aggregated_features[name].iloc[2] == np.mean([2, 4])
    assert hg._aggregated_features[name].iloc[3] == np.mean([2, 32])
    assert hg._aggregated_features[name].iloc[4] == np.mean([16])

    name = hg.aggregate("f1", 2, np.mean)
    assert hg._aggregated_features[name].iloc[0] == np.mean([32])
    assert hg._aggregated_features[name].iloc[1] == np.mean([16])
    assert hg._aggregated_features[name].iloc[2] == np.mean([16])
    assert hg._aggregated_features[name].iloc[3] == np.mean([4, 8])
    assert hg._aggregated_features[name].iloc[4] == np.mean([2])

    name = hg.aggregate("f1", 3, np.mean)
    assert hg._aggregated_features[name].iloc[0] == np.mean([0])
    assert hg._aggregated_features[name].iloc[1] == np.mean([32])
    assert hg._aggregated_features[name].iloc[2] == np.mean([32])
    assert hg._aggregated_features[name].iloc[3] == np.mean([0])
    assert hg._aggregated_features[name].iloc[4] == np.mean([4, 8])


def test_add_features_from_csv() -> None:
    """test add_features_from_csv with and without header"""
    hg_01 = Mesh2VecBase.from_file(Path("data/hyper_02.txt"), 3)
    hg_01.add_features_from_csv(Path("data/hyper_02_features.csv"), with_header=True)

    hg_02 = Mesh2VecBase.from_file(Path("data/hyper_02.txt"), 3)
    hg_02.add_features_from_csv(
        Path("data/hyper_02_features_noheader.csv"),
        columns=["vtx_id", "by_2", "by_3", "pow2", "sqrt"],
    )

    hg_03 = Mesh2VecBase.from_file(Path("data/hyper_02.txt"), 3)
    hg_03.add_features_from_csv(
        Path("data/hyper_02_features.csv"),
        columns=["vtx_id", "by_2", "by_3", "pow2", "sqrt!"],
    )

    assert all(hg_01._features == hg_02._features) and all(
        hg_02._features == hg_03._features.rename({"sqrt!": "sqrt"}, axis=1)
    )


def test_add_features_from_dataframe() -> None:
    """test add_features_from_dataframe"""
    edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
    hg = Mesh2VecBase(3, edges)
    df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": ["a", "b", "c", "x", "y"]})
    hg.add_features_from_dataframe(df1)
    assert hg._features["f1"].tolist() == ["a", "b", "c", "x", "y"]

    df2 = pd.DataFrame({"vtx_id": ["x", "y", "a", "b", "c"], "f2": ["x", "y", "a", "b", "c"]})
    hg.add_features_from_dataframe(df2)
    assert hg._features["f2"].tolist() == ["a", "b", "c", "x", "y"]

    df3 = pd.DataFrame({"vtx_id": ["x"], "f3": ["x"]})
    hg.add_features_from_dataframe(df3)
    assert np.isnan(hg._features["f3"][1])
    assert hg._features["f3"][3] == "x"
