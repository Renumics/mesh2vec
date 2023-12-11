"""Mesh2VecCae"""
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, DefaultDict, Optional, Callable, Union, Tuple, Dict, Any

import numpy as np
import pandas as pd
import trimesh
from lasso.dyna import D3plot, ArrayType
import plotly.graph_objects as go

from mesh2vec import mesh_features
from mesh2vec.mesh_features import CaeShellMesh, is_tri, num_border, midpoint
from mesh2vec.mesh2vec_base import Mesh2VecBase
from mesh2vec.mesh2vec_exceptions import (
    check_feature_available,
    AnsaNotFoundException,
    check_vtx_id_match,
)


class Mesh2VecCae(Mesh2VecBase):
    """
    Class to use finite element meshes as hypergraphs. Provide methods to add features
    from common CAE tools (for now: ANSA meshes, LSDYNA binout and d3plot information
    from lasso-cae) Inherits from Mesh2VecBase

    All vertices in this class are assumed to be elements in CAE meshes
    (restricted to LSDYNA so far).
    """

    def __init__(
        self,
        distance: int,
        mesh: CaeShellMesh,
        mesh_info: pd.DataFrame,
        calc_strategy: str = "dfs",
    ) -> None:
        # pylint: disable=line-too-long
        """
        Create a Mesh2VecCae instance

        Args:
            distance: the maximum distance for neighborhood generation and feature aggregation
            mesh: points, point_ids/uids, connectivity and element_ids/uids
            mesh_info: additional info about the elements in mesh (same order is required)
                columns "part_name", "part_id", "file_path", "element_id" are required
                calc_strategy: choose the algorithm to calculate adjacencies

                * "dfs": depth first search (defaultl fast)
                * "bfs": breadth first search (low memory consumption)
                * "matmul": matrix multiplication (deprecated, for compatibility only)

        Example:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from mesh2vec.mesh2vec_cae import Mesh2VecCae
            >>> from mesh2vec.mesh_features import CaeShellMesh
            >>> point_coordinates = np.array([[v, v, v] for v in range(6)])
            >>> pnt_ids = np.array(["0", "1", "2", "3", "4", "5"])
            >>> elem_ids = np.array(["0", "1", "2", "3"])
            >>> elem_node_idxs = np.array([[0, 1, 2, 2], [1, 2, 3, 3], [2, 3, 4, 4], [3, 4, 5, 5]])
            >>> mesh = CaeShellMesh(point_coordinates, pnt_ids, elem_ids, elem_node_idxs)
            >>> mesh_info = pd.DataFrame({"element_id": elem_ids})
            >>> mesh_info["part_name"] = "part_name"
            >>> mesh_info["part_id"] = "part_id"
            >>> mesh_info["file_path"] = "file_path"
            >>> m2v = Mesh2VecCae(2, mesh, mesh_info)
            >>> m2v._hyper_edges
            OrderedDict([('0', ['0']), ('1', ['0', '1']), ('2', ['0', '1', '2']), ('3', ['1', '2', '3']), ('4', ['2', '3']), ('5', ['3'])])
        """
        assert len(mesh.element_node_idxs) == len(mesh.element_ids)
        assert len(mesh.point_ids) == len(mesh.point_coordinates)
        assert mesh.element_ids.dtype.type is np.str_
        assert mesh.point_ids.dtype.type is np.str_

        points_to_faces: DefaultDict[int, List[int]] = defaultdict(list)
        for face_idx, face_pts in enumerate(mesh.element_node_idxs):
            points_to_faces[face_pts[0]].append(face_idx)
            points_to_faces[face_pts[1]].append(face_idx)
            points_to_faces[face_pts[2]].append(face_idx)
            if face_pts[2] != face_pts[3]:
                points_to_faces[face_pts[3]].append(face_idx)

        points_to_faces_str = {
            mesh.point_uid[pnt_idx]: mesh.element_uid[faces_idxs].tolist()
            for pnt_idx, faces_idxs in points_to_faces.items()
        }

        hyper_vtx_ids = mesh.element_uid

        super().__init__(
            distance,
            points_to_faces_str,
            vtx_ids=hyper_vtx_ids.tolist(),
            calc_strategy=calc_strategy,
        )
        self._mesh = mesh
        self._element_info = pd.DataFrame(
            {
                "vtx_id": self._vtx_ids_to_idx.keys(),
                "element_id": mesh.element_ids,
            }
        )

        assert all(x in mesh_info.keys() for x in ["file_path", "element_id"])
        assert len(mesh_info) == len(self._element_info)
        assert all(mesh_info["element_id"] == self._element_info["element_id"])
        mesh_info = mesh_info.drop(["element_id"], axis=1)

        self._element_info = pd.concat([self._element_info, mesh_info], axis=1)

    @staticmethod
    def from_stl_shell(distance: int, stlfile: Path) -> "Mesh2VecCae":
        """
        Read the given stl file and use the shell elements to generate a hypergraph, using mesh
        nodes as hyperedges, and adjacent elements as  hypervertices.
        """
        trimesh_mesh: trimesh.Trimesh = trimesh.load(stlfile)
        mesh = CaeShellMesh.from_trimesh(trimesh_mesh)

        element_info = pd.DataFrame({"element_id": mesh.element_ids})
        element_info["part_name"] = "stl_shell"
        element_info["part_id"] = "stl_shell"
        element_info["file_path"] = str(stlfile)

        return Mesh2VecCae(distance, mesh, element_info)

    # pylint: disable=too-many-arguments
    @staticmethod
    def from_ansa_shell(
        distance: int,
        ansafile: Path,
        partid: str = "",
        json_mesh_file: Optional[Path] = None,
        ansa_executable: Optional[Path] = None,
        ansa_script: Optional[Path] = None,
        verbose: bool = False,
        calc_strategy: str = "dfs",
    ) -> "Mesh2VecCae":
        """
        Read the given ANSA file and use the shell elements corresponding with ``partid`` to
        generate a hypergraph, using CAE nodes as hyperedges, and adjacent elements as
        hypervertices. This is a simple wrapper around an ANSA Python script that generates
        a json_mesh_file, if ``json_mesh_file`` is a valid file path. Otherwise, a temporary
        file is generated and deleted after importing. ``json_mesh_file`` is not overwritten
        if it already exists .
        For now, this function is only guaranteed
        to work with shell meshes for LSDYNA. Each element gets a unique internal ID consisting
        of its element ID (which may be not unique), and a unique hash value.


        Args:
            distance: the maximum distance for neighborhood generation and feature aggregation
            ansafile: path to ansa file
            partid: part id to use for hypergraph generation
            json_mesh_file: path to json mesh file,
                if exists, it will be loaded instead of the ansafile else it will be generated.
            ansa_executable: path to ansa executable
                Path to ANSA executable can also be provided in environment var: ANSA_EXECUTABLE
            ansa_script: path to ansa script
                You can use a customized script to include more features ansa_script
                (see :ref:`Customize Ansa script<customize_ansa_script>`)
            verbose: print additional information
            calc_strategy: choose the algorithm to calculate adjacencies

                * "dfs": depth first search (defaultl fast)
                * "bfs": breadth first search (low memory consumption)
                * "matmul": matrix multiplication (deprecated, for compatibility only)

        Example:
            >>> from pathlib import Path
            >>> from mesh2vec.mesh2vec_cae import Mesh2VecCae
            >>> m2v = Mesh2VecCae.from_ansa_shell(4,
            ...     Path("data/hat/Hatprofile.k"),
            ...     json_mesh_file=Path("data/hat/cached_hat_key.json"))
            >>> len(m2v._hyper_edges)
            6666
        """
        if ansa_executable is not None:
            os.environ["ANSA_EXECUTABLE"] = str(ansa_executable)

        elements, nodes = Mesh2VecCae._read_ansafile(
            ansafile, json_mesh_file, verbose=verbose, partid=partid, ansa_script=ansa_script
        )
        mesh = CaeShellMesh.from_ansa_json(elements, nodes)

        element_info = pd.DataFrame(
            {
                "element_id": mesh.element_ids.tolist(),
                **{k: [e[k] for e in elements] for k in ["PID", "part_name"]},
            }
        )
        element_info["part_id"] = [element["__part__"] for element in elements]
        element_info["file_path"] = str(ansafile)
        return Mesh2VecCae(distance, mesh, element_info, calc_strategy)

    @staticmethod
    def from_d3plot_shell(
        distance: int, d3plot: Path, partid: str = None, calc_strategy="dfs"
    ) -> "Mesh2VecCae":
        """
        Read the given d3plot file and use the shell elements corresponding with ``partid`` to
        generate a hypergraph, using CAE nodes as hyperedges, and adjacent elements as
        hypervertices.

        Args:
            distance: the maximum distance for neighborhood generation and feature aggregation
            d3plot: path to d3plot file
            partid: part id to use
            calc_strategy: choose the algorithm to calculate adjacencies

                * "dfs": depth first search (defaultl fast)
                * "bfs": breadth first search (low memory consumption)
                * "matmul": matrix multiplication (deprecated, for compatibility only)

        Example:
            >>> from pathlib import Path
            >>> from mesh2vec.mesh2vec_cae import Mesh2VecCae
            >>> m2v = Mesh2VecCae.from_d3plot_shell(3, Path("data/hat/HAT.d3plot"))
            >>> len(m2v._hyper_edges)
            6666
        """
        d3plot_data = D3plot(d3plot.as_posix())
        mesh = CaeShellMesh.from_d3plot(d3plot_data, int(partid) if partid is not None else None)

        if partid is not None:
            part_index = np.where(d3plot_data.arrays["part_ids"] == int(partid))[0]
            selected_element_indexes = np.where(
                d3plot_data.arrays[ArrayType.element_shell_part_indexes] == part_index
            )[0]

        else:
            selected_element_indexes = np.array(
                range(len(d3plot_data.arrays[ArrayType.element_shell_part_indexes]))
            )

        element_info = pd.DataFrame({"element_id": mesh.element_ids})
        selected_elements_part_indexes = d3plot_data.arrays[ArrayType.element_shell_part_indexes][
            selected_element_indexes
        ]
        element_info["part_id"] = [
            str(v) for v in d3plot_data.arrays[ArrayType.part_ids][selected_elements_part_indexes]
        ]
        element_info["part_name"] = d3plot_data.arrays[ArrayType.part_ids][
            selected_elements_part_indexes
        ]
        element_info["file_path"] = str(d3plot)

        return Mesh2VecCae(distance, mesh, element_info, calc_strategy=calc_strategy)

    @staticmethod
    def from_keyfile_shell(distance: int, keyfile: Path, calc_strategy="bfs") -> "Mesh2VecCae":
        """
        Read the given keyfile and use the shell elements to generate a hypergraph, using mesh
        nodes as hyperedges, and adjacent elements as hypervertices.

        Args:
            distance: the maximum distance for neighborhood generation and feature aggregation
            keyfile: path to keyfile
            calc_strategy: choose the algorithm to calculate adjacencies

                * "dfs": depth first search (defaultl fast)
                * "bfs": breadth first search (low memory consumption)
                * "matmul": matrix multiplication (deprecated, for compatibility only)

        Example:
            >>> from pathlib import Path
            >>> from mesh2vec.mesh2vec_cae import Mesh2VecCae
            >>> m2v = Mesh2VecCae.from_keyfile_shell(4, Path("data/hat/Hatprofile.k"))
            >>> len(m2v._hyper_edges)
            6666
        """
        mesh = CaeShellMesh.from_keyfile(keyfile)
        element_info = pd.DataFrame({"element_id": mesh.element_ids})
        element_info["file_path"] = str(keyfile)
        return Mesh2VecCae(distance, mesh, element_info, calc_strategy=calc_strategy)

    def get_elements_info(self) -> pd.DataFrame:
        """
        Return a Pandas dataframe containing a row for each element in the hypergraph
        vertices with

        * its internal ID (vtx_id)
        * element ID (element_id)
        * part ID (part_id)
        * Part name, or None if not available (part_name)
        * File path, or None if not available (file_path)
        """
        return self._element_info.copy()

    # pylint: disable=too-many-arguments
    def add_features_from_ansa(
        self,
        features: List[str],
        ansafile: Optional[Path] = None,
        json_mesh_file: Optional[Path] = None,
        ansa_executable: Optional[Path] = None,
        ansa_script: Optional[Path] = None,
        verbose: bool = False,
        allow_addtional_ansa_features: bool = False,
    ) -> List[str]:
        """
        Add values derived or calculated from ANSA shell elements (currently restricted to
        LSDYNA models) for each element.

        Path to ANSA executable can also be provided in environment var: ANSA_EXECUTABLE

        You can use a customized script to include more features ansa_script
        (see :ref:`Customize Ansa script<customize_ansa_script>`)


        ``features`` is a subset of:

            * aspect: The aspect ratio of each element (ansafile is required)
            * warpage: (ansafile is required)
            * num_border: number of border edges
            * is_tria
            * midpoint x,y,z
            * normal vector x,y,z (ansafile is required)
            * area (ansafile is required)
            * custom features from your customized ansa_script (scalar values only,
                allow_addtional_ansa_features must be True)

        Example:
            >>> from pathlib import Path
            >>> from mesh2vec.mesh2vec_cae import Mesh2VecCae
            >>> m2v = Mesh2VecCae.from_ansa_shell(
            ...    4,
            ...    Path("data/hat/Hatprofile.k"),
            ...    json_mesh_file=Path("data/hat/cached_hat_key.json"))
            >>> m2v.add_features_from_ansa(
            ...    ["aspect", "warpage"],
            ...    Path("data/hat/Hatprofile.k"),
            ...    json_mesh_file=Path("data/hat/cached_hat_key.json"))
            ['aspect', 'warpage']
            >>> print(f'{m2v._features["warpage"][14]:.4f}')
            0.0188
        """
        # pylint: disable=too-many-locals
        okay_ansa = ["aspect", "warpage", "normal", "area", "skew"]
        okay_inplace = ["num_border", "is_tria", "midpoint"]

        for feature in features:
            if not feature in okay_ansa + okay_inplace:
                if not allow_addtional_ansa_features:
                    okay_ansa.append(feature)
                else:
                    raise ValueError(
                        f"Feature {feature} is unknown and allow_addtional_ansa_features is False"
                        f"All features must be in {okay_ansa+okay_inplace}"
                    )

        if ansa_executable is not None:
            os.environ["ANSA_EXECUTABLE"] = str(ansa_executable)

        if any(feature in okay_ansa for feature in features):
            elements, nodes = Mesh2VecCae._read_ansafile(
                ansafile, json_mesh_file, verbose=verbose, ansa_script=ansa_script
            )
            mesh = CaeShellMesh.from_ansa_json(elements, nodes)

            def _err_to_nan(elements: List[Dict[Any, Any]], name: str) -> np.ndarray:
                return np.array(
                    [np.nan if e[name] == "error" else e[name] for e in elements],
                    dtype=float,
                )

            element_metrics = pd.DataFrame(
                {
                    "vtx_id": mesh.element_uid.tolist(),
                    **{k: _err_to_nan(elements, k) for k in okay_ansa if k != "normal"},
                    "normal": [e["normal"] for e in elements],
                }
            )
            self.add_features_from_dataframe(
                element_metrics[
                    ["vtx_id"] + [feature for feature in features if feature in okay_ansa]
                ]
            )

        if any(feature in okay_inplace for feature in features):
            element_metrics = pd.DataFrame(
                {
                    "vtx_id": self._mesh.element_uid.tolist(),
                    "is_tria": is_tri(self._mesh.element_node_idxs),
                    "midpoint": [
                        list(v)
                        for v in midpoint(
                            self._mesh.element_node_idxs, self._mesh.point_coordinates
                        )
                    ],
                    "num_border": num_border(self._mesh.element_node_idxs),
                }
            )
            self.add_features_from_dataframe(
                element_metrics[
                    ["vtx_id"] + [feature for feature in features if feature in okay_inplace]
                ]
            )
        return features

    # pylint: disable=too-many-arguments, too-many-branches. too-many-statements
    def get_feature_from_d3plot(
        self,
        feature: str,
        d3plot_data: D3plot,
        timestep: int = None,
        shell_layer: Union[int, Callable] = None,
        history_var_index: int = None,
    ) -> Tuple[str, List[Any]]:
        """
        Map a single feature from a d3plot to the CAE elements which are the vertices
        of the hg. Restricted to arrays available from lasso-cae.

        Example:
            >>> from pathlib import Path
            >>> from lasso.dyna import ArrayType, D3plot
            >>> from mesh2vec.mesh2vec_cae import Mesh2VecCae
            >>> m2v =  Mesh2VecCae.from_d3plot_shell(3, Path("data/hat/HAT.d3plot"))
            >>> names, values, ids = m2v.get_feature_from_d3plot(
            ...    ArrayType.element_shell_strain,
            ...    D3plot(Path("data/hat/HAT.d3plot").as_posix()),
            ...    timestep=1, shell_layer=0)
            >>> print([f'{v:.4f}' for v in values[42]])
            ['0.0010', '-0.0003', '-0.0000', '-0.0012', '-0.0000', '-0.0003']
        """

        def _get_d3plot_layer_array_names() -> List[str]:
            return [
                ArrayType.element_shell_stress,
                ArrayType.element_shell_effective_plastic_strain,
                ArrayType.element_shell_history_vars,
            ]

        # validate
        okay = [
            name for name in ArrayType.__dict__ if "element_shell" in name and not "__" in name
        ]

        if not feature in okay:
            raise ValueError(
                f"Feature {feature} is unknown. features argument must be one of {okay}."
            )
        if feature in ArrayType.get_state_array_names():
            if timestep is None:
                raise ValueError(
                    f"timestep argument must be set for requested Feature '{feature}'."
                )
        if feature in _get_d3plot_layer_array_names():
            if shell_layer is None:
                raise ValueError(
                    f"shell_layer argument must be set for requested Feature '{feature}'."
                )

        if feature == ArrayType.element_shell_history_vars:
            if history_var_index is None:
                raise ValueError(
                    f"history_var_index argument must be set for requested Feature '{feature}'."
                )

        # build name
        if timestep is None:
            timestep_suffix = ""
        else:
            timestep_suffix = f"_{timestep}"
        if shell_layer is None:
            shell_layer_suffix = ""
        else:
            if callable(shell_layer):
                shell_layer_suffix = f"_{shell_layer.__name__}"
            else:
                shell_layer_suffix = f"_{shell_layer}"
        if history_var_index is None:
            history_var_suffix = ""
        else:
            history_var_suffix = f"_{history_var_index}"

        # get values
        if feature in ArrayType.get_state_array_names():
            if feature in [
                ArrayType.element_shell_stress,
                ArrayType.element_shell_strain,
            ]:
                if callable(shell_layer):
                    new_feature = np.squeeze(d3plot_data.arrays[feature][timestep, :, :, :])
                    new_feature = [shell_layer(x).tolist() for x in new_feature]
                else:
                    new_feature = [
                        v.tolist()
                        for v in np.squeeze(
                            d3plot_data.arrays[feature][timestep, :, shell_layer, :]
                        )
                    ]
                feature_name = feature + timestep_suffix + shell_layer_suffix

            elif feature == ArrayType.element_shell_history_vars:
                if callable(shell_layer):
                    new_feature = d3plot_data.arrays[feature][timestep, :, :, history_var_index]
                    new_feature = [shell_layer(x) for x in new_feature]
                else:
                    new_feature = d3plot_data.arrays[feature][
                        timestep, :, shell_layer, history_var_index
                    ]
                feature_name = feature + timestep_suffix + shell_layer_suffix + history_var_suffix

            elif feature == ArrayType.element_shell_effective_plastic_strain:
                if callable(shell_layer):
                    new_feature = np.squeeze(d3plot_data.arrays[feature][timestep, :, :]).tolist()
                    new_feature = [shell_layer(x) for x in new_feature]
                else:
                    new_feature = np.squeeze(
                        d3plot_data.arrays[feature][timestep, :, shell_layer]
                    ).tolist()
                feature_name = feature + timestep_suffix + shell_layer_suffix

            else:
                new_feature = [
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in d3plot_data.arrays[feature][timestep, :]
                ]
                feature_name = feature + timestep_suffix
        else:
            new_feature = [
                v.tolist() if isinstance(v, np.ndarray) else v
                for v in d3plot_data.arrays[feature]
            ]
            feature_name = feature

        return (
            feature_name,
            new_feature,
            [str(x) for x in d3plot_data.arrays["element_shell_ids"]],
        )

    def add_feature_from_d3plot(
        self,
        feature: str,
        d3plot: Union[Path, D3plot],
        timestep: int = None,
        shell_layer: Union[int, Callable] = None,
        history_var_index: int = None,
    ) -> str:
        """
        Map feature from a d3plot to the CAE elements which are the vertices
        of the hg. Restricted to arrays available from lasso-cae.

        Args:
            feature: name of the feature to add (a shell_array name of lasso.dyna.ArrayType)
            d3plot: path to d3plot file or loaded d3plot data
            timestep: timestep to extract (required for time dependend arrays, ignored otherwise)
            shell_layer: integration point index or callabe to accumulate over all integration
                points (required for layerd arrays, ignored otherwise)
            history_var_index: index of the history variable to extract (required for
                element_shell_history_vars, ignored otherwise)


        Example:
            >>> from pathlib import Path
            >>> from lasso.dyna import ArrayType
            >>> from mesh2vec.mesh2vec_cae import Mesh2VecCae
            >>> m2v =  Mesh2VecCae.from_d3plot_shell(3, Path("data/hat/HAT.d3plot"))
            >>> name = m2v.add_feature_from_d3plot(
            ...    ArrayType.element_shell_strain,
            ...    Path("data/hat/HAT.d3plot"),
            ...    timestep=1, shell_layer=0)
            >>> # print eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz of layer 0
            >>> print([f'{v:.4f}' for v in m2v.features()[name][42]])
            ['0.0010', '-0.0003', '-0.0000', '-0.0012', '-0.0000', '-0.0003']

        Example:
            >>> from pathlib import Path
            >>> from lasso.dyna import ArrayType
            >>> from mesh2vec.mesh2vec_cae import Mesh2VecCae
            >>> m2v =  Mesh2VecCae.from_d3plot_shell(3, Path("data/hat/HAT.d3plot"))
            >>> def mean_over_components_all_layers(v):
            ...    return np.mean(v, axis=-1)
            >>> name = m2v.add_feature_from_d3plot(
            ...    ArrayType.element_shell_strain,
            ...    Path("data/hat/HAT.d3plot"),
            ...    timestep=-1, shell_layer=mean_over_components_all_layers)
            >>> # print mean of all components for each layer (0,1)
            >>> print([f'{v:.4f}' for v in m2v.features()[name][42]])
            ['0.0002', '0.0001']
        """

        d3plot_data = D3plot(d3plot.as_posix()) if not isinstance(d3plot, D3plot) else d3plot

        feature_name, feature_values, element_shell_ids = self.get_feature_from_d3plot(
            feature=feature,
            d3plot_data=d3plot_data,
            timestep=timestep,
            shell_layer=shell_layer,
            history_var_index=history_var_index,
        )

        check_vtx_id_match(self._mesh.element_ids, element_shell_ids)

        new_features = pd.DataFrame({"vtx_id": element_shell_ids})
        new_features[feature_name] = feature_values
        self._features = self._features.merge(  # type: ignore
            new_features, how="left", on="vtx_id", validate="1:1"
        )
        return feature_name

    # pylint:disable=too-many-arguments
    def aggregate_angle_diff(
        self,
        dist: Union[List[int], int],
        aggr: Optional[Callable] = None,
        agg_add_ref: bool = True,
        default_value: float = 0.0,
        skip_arcos: bool = False,
    ) -> Union[str, List[str]]:
        # pylint: disable=line-too-long

        """
        Aggregate angle differences

        Aggregate a new feature calculated from the angle difference of each element's normal vector
        to the reference vector given by the center element  (in radian).

        Args:
            dist: either

                * distance of the maximum neighborhood of vertex for aggregation, 0 <= ``dist`` <=  ``self.distance``, or
                * a list of distances, e.g. ``[0, 2, 5]``.
            aggr: aggregation function, default is np.mean
            agg_add_ref: the aggregation callable needs the feature value of the center element as
                reference as 2nd argument. (default is True)
            default_value:  value to use in aggregation when a feature is missing for a neighbor
                or no neighor with the given dist exist.
            skip_arcos: if True, the angle difference is not calculated, but the dot product of the normal vectors is used.

        Example:
            >>> from pathlib import Path
            >>> from lasso.dyna import ArrayType
            >>> from mesh2vec.mesh2vec_cae import Mesh2VecCae
            >>> m2v = Mesh2VecCae.from_ansa_shell(
            ...    4,
            ...    Path("data/hat/Hatprofile.k"),
            ...    json_mesh_file=Path("data/hat/cached_hat_key.json"))
            >>> m2v.add_features_from_ansa(
            ...    ["normal"],
            ...    Path("data/hat/Hatprofile.k"),
            ...    json_mesh_file=Path("data/hat/cached_hat_key.json"))
            ['normal']
            >>> name = m2v.aggregate_angle_diff(2)
            >>> print(f'{ m2v._aggregated_features[name][14]:.4f}')
            0.6275


        """
        check_feature_available("normal", self)

        if aggr is None:
            aggr = np.mean

        def _mean_dir_diff(values: List[List[float]], ref_value: List[float]) -> float:
            """direction difference of values to ref_value (in radian)"""

            def vectorize_angle_diff(values, ref_value):
                try:
                    values_array = np.array(values.tolist())
                except AttributeError:
                    values_array = np.array(values)

                ref_value_array = np.array(ref_value)
                dot_products = np.dot(values_array, ref_value_array)
                if skip_arcos:
                    return dot_products
                angle_diff = np.arccos(np.clip(dot_products, -1.0, 1.0))
                return angle_diff

            angle_diff_b = vectorize_angle_diff(values, ref_value)

            return aggr(angle_diff_b)

        return self.aggregate(
            "normal",
            dist,
            aggr=_mean_dir_diff,
            aggr_name=aggr.__name__,
            agg_add_ref=agg_add_ref,
            default_value=default_value,
        )

    def get_visualization_trimesh(self, feature: str) -> trimesh.Trimesh:
        """
        Get a trimesh with face colored by feature values
        Use trimesh_mesh.show(smooth=False, flags={"cull": False}) to visualize.
        """

        df = self._element_info.merge(self._aggregated_features[["vtx_id", feature]], on="vtx_id")
        assert len(df) == len(self._mesh.element_node_idxs)
        feature_values = df[feature]
        element_node_idxs = self._mesh.element_node_idxs

        # quads to triangles (element and feature)
        tri_faces, tri_features = mesh_features.quads_to_tris_feature_list(
            element_node_idxs, feature_values
        )

        omin = tri_features.min()
        omax = tri_features.max()
        tri_features_scaled = (tri_features - omin) / (omax - omin)

        trimeh_mesh = trimesh.Trimesh(
            vertices=self._mesh.point_coordinates / np.max(self._mesh.point_coordinates),
            faces=tri_faces,
        )

        trimeh_mesh.visual.face_colors = np.array(
            [np.array([254 - 254 * x, 254 * x, 0, 254]) for x in tri_features_scaled]
        )

        return trimeh_mesh

    def get_visualization_plotly(self, feature: str) -> go.Figure:
        """
        visualize an aggregated feature on mesh
        Useage in Notebook
        import plotly.io as pio
        pio.renderers.default = 'sphinx_gallery'
        fig = m2v.get_visualization_plotly(name)
        fig
        """
        df = self._element_info.merge(self._aggregated_features[["vtx_id", feature]], on="vtx_id")
        assert len(df) == len(self._mesh.element_node_idxs)
        feature_values = df[feature].tolist()

        def _wireframe() -> go.Scatter3d:
            pts = self._mesh.point_coordinates[self._mesh.element_node_idxs]
            nans = np.empty(pts[:, [1], :].shape)
            nans[:] = np.nan
            pts = np.concatenate([pts, pts[:, [0], :], nans], axis=1)
            tri_points = pts.reshape([-1, 3])
            x, y, z = tri_points.T
            lines = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": "rgb(80,80,80)", "width": 5},
            )
            return lines

        def _faces(feature: str, feature_values: List[Any]) -> go.Mesh3d:
            element_node_idxs = self._mesh.element_node_idxs
            tri_faces, tri_features = mesh_features.quads_to_tris_feature_list(
                element_node_idxs, feature_values
            )
            go_mesh = go.Mesh3d(
                x=self._mesh.point_coordinates[:, 0],
                y=self._mesh.point_coordinates[:, 1],
                z=self._mesh.point_coordinates[:, 2],
                colorbar_title=feature,
                colorscale=[[0, "gold"], [0.5, "mediumturquoise"], [1, "magenta"]],
                intensity=tri_features,
                intensitymode="cell",
                i=tri_faces[:, 0],
                j=tri_faces[:, 1],
                k=tri_faces[:, 2],
                showscale=True,
            )
            return go_mesh

        layout = go.Layout(scene={"aspectmode": "data"})
        fig = go.Figure(data=[_faces(feature, feature_values), _wireframe()], layout=layout)

        return fig

    @staticmethod
    def _read_ansafile(
        ansafile: Optional[Path],
        json_mesh_file: Optional[Path],
        verbose: bool,
        partid: str = "",
        ansa_script: Optional[Path] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        ansa_path = os.getenv("ANSA_EXECUTABLE")

        if ansa_script is not None:
            if not ansa_script.exists():
                raise FileNotFoundError(
                    f"Ansa Script was not found at '{ansa_script}' from current "
                    f"directory '{os.getcwd() }'"
                )
        else:
            src_folder = os.path.abspath(os.path.dirname(__file__))
            ansa_script = f"{src_folder}/templates/ansa.py"

        with TemporaryDirectory() as tmp_folder:
            if json_mesh_file is None:
                output_path = Path(tmp_folder) / "tmp.json"
            else:
                output_path = json_mesh_file
            if not output_path.exists():
                if ansa_path is None or not Path(ansa_path).exists():
                    raise AnsaNotFoundException(
                        f"Ansa was not found at '{ansa_path}'. "
                        "Make sure the environment variable "
                        "ANSA_EXECUTABLE contains the path "
                        "to the Ansa executable."
                    )
                command = [
                    f"{ansa_path}",
                    "-b",
                    "-foregr",
                    "-execpy",
                    f"load_script: '{ansa_script}",
                    "-execpy",
                    f"make_hg('{ansafile}', '{output_path}', '{partid}')",
                ]
                if verbose:
                    subprocess.check_call(command)
                else:
                    subprocess.check_call(command, stdout=subprocess.DEVNULL)
            data = json.loads(output_path.read_text())

        elements, nodes = data["elements"], data["nodes"]
        return elements, nodes

    def mesh(self) -> CaeShellMesh:
        """
        return the FE mesh
        """
        return self._mesh
