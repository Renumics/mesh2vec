"""Mesh2VecBase"""
import collections
from pathlib import Path
from typing import List, Optional, Callable, OrderedDict, Dict, Union, Iterable

import networkx
import numpy as np
import pandas as pd
from scipy.sparse import csr_array

# noinspection PyProtectedMember
from pandas.api.types import is_string_dtype

from mesh2vec.helpers import calc_adjacencies
from mesh2vec.mesh2vec_exceptions import (
    check_distance_init_arg,
    check_distance_arg,
    check_hyper_edges,
    check_vtx_ids,
    check_vtx_arg,
    check_feature_available,
    check_vtx_ids_column,
)


class Mesh2VecBase:
    # pylint: disable=line-too-long,too-many-instance-attributes
    """
    Class to derive hypergraph neighborhoods from ordinary, undirected graphs, map numerical
    features to vertices, and provide aggregation methods that result in fixed sized vectors
    suitable for machine learning methods.
    """

    def __init__(
        self,
        distance: int,
        hyper_edges: Dict[str, List[str]],
        vtx_ids: Optional[List[str]] = None,
    ):
        r"""
        Create neighborhood sets on a hypergraph.
        For each vertex in the  hg, create sets of neighbors in distance d for each d up to a given
        max distance. A neighborhood set :math:`N^d_v` of vertex :math:`v` in distance :math:`d`
        contains all vertices :math:`w` where :math:`dist(v,w) \leq d` and
        not :math:`w \in N^\delta_v,\delta <  d`.

        Args:
            distance: the maximum distance for neighborhood generation and feature aggregation
            hyper_edges: edge->connected vertices
            vtx_ids: provide a list of all vertices to control inernal order of vertices
                (features, aggregated feature)

        Example:
            >>> from mesh2vec.mesh2vec_base import Mesh2VecBase
            >>> edges = {"first": ["a", "b", "c"], "second": ["x", "y"]}
            >>> hg = Mesh2VecBase(3, edges)
            >>> hg._hyper_edges
            OrderedDict([('first', ['a', 'b', 'c']), ('second', ['x', 'y'])])

        """
        check_distance_init_arg(distance)
        check_hyper_edges(hyper_edges)

        if vtx_ids is None:
            vtx_ids = np.unique(
                [vtx_id for vtx_ids in hyper_edges.values() for vtx_id in vtx_ids]
            ).tolist()
        check_vtx_ids(vtx_ids, hyper_edges)

        self._distance: int = distance
        self._hyper_edges: OrderedDict[str, List[str]] = collections.OrderedDict(hyper_edges)

        self._vtx_idx_to_ids = collections.OrderedDict(enumerate(vtx_ids))  # type: ignore
        self._vtx_ids_to_idx = {vtx_ids: i for i, vtx_ids in enumerate(vtx_ids)}

        self._features: pd.DataFrame = pd.DataFrame({"vtx_id": self._vtx_ids_to_idx.keys()})
        self._aggregated_features: pd.DataFrame = pd.DataFrame(
            {"vtx_id": self._vtx_ids_to_idx.keys()}
        )

        hyper_edges_idx = collections.OrderedDict(
            (
                h_edge_id,
                [self._vtx_ids_to_idx[vtx_id] for vtx_id in vtx_ids],
            )
            for h_edge_id, vtx_ids in self._hyper_edges.items()
        )

        self._adjacency_matrix_powers, self._adjacency_matrix_powers_exclusive = calc_adjacencies(
            hyper_edges_idx, distance
        )

    @staticmethod
    def from_file(hg_file: Path, distance: int) -> "Mesh2VecBase":
        # pylint: disable=line-too-long
        r"""
        Read a hypergraph (hg) from a text file.

        Args:
            hg_file: either

                * a CSV files of pairs of alphanumerical vertex identifiers defining an undirected graph. Multiple edges are ignored. The initial hypergraph is given by the cliques of the graph. Since the CLIQUE problem is NP-complete, use this for small graphs only.
                * a hypergraph file (text). Each line of the file contains an alphanumerical edge identifier, followed by a list of vertex identifiers the edge is containing, in the form 'DGEID: VTXID1,VTXID2,...'
            distance: the maximum distance for neighborhood generation and feature aggregation

        Example:
            >>> from pathlib import Path
            >>> from mesh2vec.mesh2vec_base import Mesh2VecBase
            >>> hg = Mesh2VecBase.from_file(Path("data/hyper_02.txt"), 4)
            >>> hg._hyper_edges["edge1"]
            ['vtx00', 'vtx01', 'vtx07', 'vtx11']

        """
        check_distance_init_arg(distance)

        if hg_file.suffix == ".csv":
            single_edges_ids = pd.read_csv(hg_file, header=0)
            single_edges_ids = single_edges_ids.astype(str)

            vtx_ids = np.unique(single_edges_ids)
            vtx_idx_to_ids = dict(enumerate(vtx_ids))  # type: ignore
            vtx_ids_to_idx = {vtx_ids: i for i, vtx_ids in enumerate(vtx_ids)}

            single_edges_idx = [
                [vtx_ids_to_idx[vtxid_a], vtx_ids_to_idx[vtxid_b]]
                for vtxid_a, vtxid_b in single_edges_ids.to_numpy()
            ]

            nx_graph = networkx.Graph(single_edges_idx)
            cliques = list(networkx.find_cliques(nx_graph))
            return Mesh2VecBase(
                distance=distance,
                hyper_edges={
                    f"clique_{i}": [vtx_idx_to_ids[v] for v in clique]
                    for i, clique in enumerate(cliques)
                },
            )

        # txt file
        with open(hg_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        lines_split = [
            line.rstrip().replace(":", ",").replace(" ", "").split(",") for line in lines
        ]
        hyper_edges_ids_to_vtx_ids = {line_split[0]: line_split[1:] for line_split in lines_split}
        return Mesh2VecBase(distance=distance, hyper_edges=hyper_edges_ids_to_vtx_ids)

    def get_nbh(self, vtx: str, dist: int) -> List[str]:
        """
        Get a list of neighbors with the exact distance ``dist`` of a given vertex ``vtx``

        Example:
            >>> from mesh2vec.mesh2vec_base import Mesh2VecBase
            >>> edges = {"first": ["a", "b", "c"], "second": ["x", "y"]}
            >>> hg = Mesh2VecBase(3, edges)
            >>> hg.get_nbh("a",1)
            ['b', 'c']
        """
        check_distance_arg(dist, self)
        check_vtx_arg(vtx, self)

        if dist == 0:
            return [vtx]

        vertices_idx = self._adjacency_matrix_powers_exclusive[dist][
            [self._vtx_ids_to_idx[vtx]], :
        ].indices
        return [self._vtx_idx_to_ids[i] for i in vertices_idx]

    def aggregate_categorical(
        self,
        feature: str,
        dist: Union[List[int], int],
        categories: Optional[Union[List[str], List[int]]] = None,
        default_value: Optional[Union[int, str]] = None,
    ) -> Union[str, List[str]]:
        """
        For categorical features, aggregate the numbers of occurrences of each categorical value.
        This results in a new aggregated ``feature`` for each categorical value. If ``feature`` is
        color and dist is 2, and there are 3 categories, e.g. RED, YELLOW, GREEN, the resulting
        feature column are named color-cat-RED-2, color-cat-YELLOW-2, color-cat-GREEN-2.
        If ``categories`` is ``None``, all unique values from ``feature`` as taken as categories,
        otherwise the categories are taken from ``categories`` which must be list-like. Values
        not existent in ``categories`` are counted in an additional category NONE.

        Returns:
            aggregated feature name(s)

        Example:
            >>> import pandas as pd
            >>> from mesh2vec.mesh2vec_base import Mesh2VecBase
            >>> edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
            >>> hg = Mesh2VecBase(3, edges)
            >>> df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"],
            ...     "f1": ["RED", "GREEN", "RED", "RED", "GREEN"]})
            >>> hg.add_features_from_dataframe(df1)
            >>> names = hg.aggregate_categorical("f1", 1)
            >>> names
            ['f1-cat-GREEN-1', 'f1-cat-RED-1']
            >>> hg._aggregated_features["f1-cat-RED-1"].to_list()
            [2, 2, 1, 1, 1]
        """
        dist_list = [dist] if isinstance(dist, int) else dist
        for dist_to_check in dist_list:
            check_distance_arg(dist_to_check, self)
        check_feature_available(feature, self)
        feature_names = []
        if categories is not None:
            feature_categories = [str(category) for category in categories] + ["NONE"]
        else:
            feature_categories = np.unique(self._features[feature])

        for distance in dist_list:
            values = self._collect_feature_values(feature, distance, default_value)

            for category in feature_categories:
                feature_name = f"{feature}-cat-{category}-{distance}"
                self._aggregated_features[feature_name] = [
                    np.sum(np.array(value) == category) for value in values
                ]
                feature_names.append(feature_name)
        if len(feature_names) == 1:
            return feature_names[0]
        return feature_names

    # pylint: disable=too-many-arguments
    def aggregate(
        self,
        feature: str,
        dist: Union[List[int], int],
        aggr: Callable,
        aggr_name: str = None,
        agg_add_ref: bool = False,
        default_value: float = 0.0,
    ) -> Union[str, List[str]]:
        # pylint: disable=line-too-long
        """
        Aggregate features from neighborhoods for each distance in ``dist``

        Args:
            feature: name of the feature to aggregate
            dist: either

                * distance of the maximum neighborhood of vertex for aggregation, 0 <= ``dist`` <=  ``self.distance``, or
                * a list of distances, e.g. ``[0, 2, 5]``.
            aggr: aggregation function, e.g. np.min, np.mean, np.median
            agg_add_ref: the aggregation callable needs the feature value of the center element as
                reference as 2nd argument.
            default_value:  value to use in aggregation when a feature is missing for a neighbor
                or no neighor with the given dist exist.
        Returns:
            aggregated feature name(s)

        The resulting aggregated features are named feature-aggr-dist for each distance in dist,
        e.g. if ``feature`` is ``aspect``, ``aggr`` is ``np.mean``, and ``dist`` is ``[0,2,3]``,
        the new aggregated features are  aspect-min-0, aspect-min-2, aspect-min-3  containing
        the mean aspect values in the hg neighborhoods in distances 0,2, and 3 for
        each vertex.

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from mesh2vec.mesh2vec_base import Mesh2VecBase
            >>> edges = {"first": ["a", "b", "c"], "second": ["x", "y"]}
            >>> hg = Mesh2VecBase(3, edges)
            >>> df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "x", "y"], "f1": [2, 4, 8, 16, 32]})
            >>> hg.add_features_from_dataframe(df1)
            >>> name = hg.aggregate("f1", 1, np.mean)
            >>> name
            'f1-mean-1'
            >>> hg._aggregated_features[name].to_list()
            [6.0, 5.0, 3.0, 32.0, 16.0]
        """
        check_feature_available(feature, self)

        dist_list = [dist] if isinstance(dist, int) else dist
        for dist_to_check in dist_list:
            check_distance_arg(dist_to_check, self)

        if aggr_name is None:
            aggr_name = aggr.__name__

        feature_names = []
        for distance in dist_list:
            values = self._collect_feature_values(feature, distance, default_value)
            if agg_add_ref:
                ref_values = self._collect_feature_values(feature, 0, default_value)
                agg_values = np.nan_to_num(
                    [aggr(value, ref_value) for value, ref_value in zip(values, ref_values)],
                    nan=default_value,
                )
            else:
                agg_values = np.nan_to_num([aggr(value) for value in values], nan=default_value)

            feature_name = f"{feature}-{aggr_name}-{distance}"
            self._aggregated_features[feature_name] = agg_values
            feature_names.append(feature_name)
        if len(feature_names) == 1:
            return feature_names[0]
        return feature_names

    def _collect_feature_values(
        self,
        feature: str,
        dist: int,
        default_value: Optional[Union[float, int, str]],
    ) -> List[List[Union[float, int, str]]]:
        """helper method to collect data from all hyper nodes during aggregation"""
        check_distance_arg(dist, self)

        if default_value is None:
            default_value = "NONE" if is_string_dtype(self._features[feature]) else np.nan
        features = self._features[feature].fillna(default_value).to_numpy()

        if dist == 0:
            return [[feature] for feature in features]

        neighborhoods = [
            self._adjacency_matrix_powers_exclusive[dist][[i], :].indices
            for i in range(len(self._features))
        ]
        values = [features[neighborhood] for neighborhood in neighborhoods]
        return values

    def add_features_from_csv(
        self,
        csv_file: Path,
        with_header: bool = False,
        columns: Optional[List[str]] = None,
    ) -> None:
        """Map the content of a CSV file to the vertices of the hypergraph.

        The column 'vtx_id' must contain the vertex IDs. if ``with_header`` is ``True``, use the
        first line as column headers. If ``columns`` is list-like, its values are taken as column
        name, overriding possible headers from the file if ``with_header`` is ``True``.
        Otherwise, all other columns starting with the 2nd are added by the name
        ``os.path.basename(csv_file).rsplit('.',1)[0]-N`` where ``N`` is the column number.

        Example:
            >>> from mesh2vec.mesh2vec_base import Mesh2VecBase
            >>> hg_01 = Mesh2VecBase.from_file(Path("data/hyper_02.txt"), 3)
            >>> hg_01.add_features_from_csv(Path("data/hyper_02_features.csv"), with_header=True)
            >>> hg_01.features()["pow2"][:4].to_list()
            [0, 1, 4, 9]
        """

        header = 0 if with_header else None
        csv_df = pd.read_csv(csv_file, header=header, names=columns)
        csv_df["vtx_id"] = csv_df["vtx_id"].astype(str)
        return self.add_features_from_dataframe(csv_df)

    def add_features_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Map the content of a Pandas dataframe to the vertices of the hypergraph. The column
        'vtx_id' of the dataframe is expected to contain vertex IDs as strings.

        Example:
            >>> from mesh2vec.mesh2vec_base import Mesh2VecBase
            >>> import pandas as pd
            >>> edges = {"first": ["a", "b", "c"], "second": ["x", "y"], "third": ["x", "a"]}
            >>> hg = Mesh2VecBase(3, edges)
            >>> df1 = pd.DataFrame({"vtx_id": ["a", "b", "c", "y"], "f1": [0, 1, 2.1, 4]})
            >>> hg.add_features_from_dataframe(df1)
            >>> hg._features["f1"].tolist()
            [0.0, 1.0, 2.1, nan, 4.0]
        """
        check_vtx_ids_column(df["vtx_id"])
        for new_columns_name in df.keys():
            if new_columns_name == "vtx_id":
                continue
            if new_columns_name in self._features:
                raise ValueError(f"Feature {new_columns_name} already exists")
        self._features = self._features.merge(df, how="left", on="vtx_id", validate="1:1")

    def to_dataframe(self, vertices: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        Returns a Pandas dataframe with all the beforehand aggregated feature columns. If
        ``vertices`` is not ``None`` and iterable, the dataframe is only generated for vertices
        in ``vertices``.
        """
        if vertices is not None:
            return self._aggregated_features[self._features in vertices].copy()
        return self._aggregated_features.copy()

    def to_array(self, vertices: Optional[Iterable[str]] = None) -> np.ndarray:
        """
        Returns a numpy array with all the beforehand aggregated feature columns. If
        ``vertices`` is not ``None`` and iterable, the array is only generated for vertices
        in ``vertices``.
        """
        return self.to_dataframe(vertices).to_numpy()

    def get_max_distance(self) -> int:
        """returns the distance value used to generate the hypergraph neighborhood"""
        return self._distance

    def available_features(self) -> List[str]:
        """returns a list the names of all features"""
        return self._features.drop("vtx_id", axis=1).keys().tolist()

    def available_aggregated_features(self) -> List[str]:
        """returns a list the names of all aggregated features"""
        return self._aggregated_features.drop("vtx_id", axis=1).keys().tolist()

    def vtx_ids(self) -> List[str]:
        """returns a list the ids of all hyper vertices"""
        return list(self._vtx_ids_to_idx.keys())

    def features(self) -> pd.DataFrame:
        """
        Returns a Pandas dataframe with all feature columns.
        """
        return self._features.copy()

    def ajacency_matrix_powers_exclusive(self) -> Dict[int, csr_array]:
        """
        Return the adjacency_matrix of different max distances
        """
        return self._adjacency_matrix_powers_exclusive
