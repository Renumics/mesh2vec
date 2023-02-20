"""Exceptions for mesh2vec"""
from typing import Any, Dict, List
from loguru import logger
import numpy as np


class InvalidDistanceArgumentException(Exception):
    """Exception raised when an invalid distance was provided"""


def check_distance_init_arg(distance: int) -> None:
    """check argument to rise exception or log warning if needed."""
    if distance < 0:
        raise InvalidDistanceArgumentException("distance must be > 0")
    if distance > 10:
        logger.warning("Calculating neighborhoods for distances > 10 can take much time")


def check_distance_arg(distance: int, hg: Any) -> None:
    """check argument to rise exception or log warning if needed."""
    if distance < 0:
        raise InvalidDistanceArgumentException("distance must be > 0")
    if distance > hg.get_max_distance():
        raise InvalidDistanceArgumentException(
            f"distance must be >= max_distance of the hypergraph: {hg.get_max_distance()}"
        )


class InvalidHyperEdgesArgument(Exception):
    """Exception raised when an invalid hyper edges dict was provided"""


def check_hyper_edges(
    hyper_edges: Dict[str, List[str]],
) -> None:
    """check argument to rise exception or log warning if needed."""
    for hyper_edge in hyper_edges:
        if not isinstance(hyper_edge, str):
            raise InvalidHyperEdgesArgument(
                "All Keys in the hyper_edges dict must be of type str"
            )
        if not all(isinstance(vtx, str) for vtx in hyper_edges[hyper_edge]):
            raise InvalidHyperEdgesArgument(
                "All values in the hyper_edges dict must be of type list[str]"
            )


class InvalidVtxIdsArgument(Exception):
    """Exception raised when an invalid vtx ids list was provided"""


def check_vtx_ids(vtx_ids: List[str], hyper_edges: Dict[str, List[str]]) -> None:
    """check argument to rise exception or log warning if needed."""
    if not all(isinstance(vtx_id, str) for vtx_id in vtx_ids):
        raise InvalidVtxIdsArgument("All vtx_ids must be of type str")

    unique_vtx_ids = np.unique([vtx_id for vtx_ids in hyper_edges.values() for vtx_id in vtx_ids])
    if not set(vtx_ids) == set(unique_vtx_ids):
        raise InvalidVtxIdsArgument(
            "The vtx_ids list must contain exactly all vertices of the hyper_edges dict."
        )


class InvalidVtxIdArgument(Exception):
    """Exception raised when an invalid vtx id was provided"""


def check_vtx_arg(vtx: str, hg: Any) -> None:
    """check argument to rise exception or log warning if needed."""
    if not isinstance(vtx, str):
        raise InvalidVtxIdArgument("vtx id must be of type str")
    if not vtx in hg.vtx_ids():
        raise InvalidVtxIdArgument(f"vtx id ({vtx}) was not found in hyper edges dict")


class FeatureDoesNotExistException(Exception):
    """Exception raised when a required feature is not available"""


def check_feature_available(feature_name: str, hg: Any) -> None:
    """check argument to rise exception or log warning if needed."""
    if not feature_name in hg.available_features():
        raise FeatureDoesNotExistException(
            f"Feature {feature_name} is not defined. "
            f"Available feature are {hg.available_features()}"
        )


class AnsaNotFoundException(Exception):
    """Exception raised when ansa executable was not found"""


class PointIdsMustBeUnqiueException(Exception):
    """Exception raised when point ids contain duplicates"""


class InvalidVtxIdsColumn(Exception):
    """Exception raised when the provided vtx_ids column contains invalid values"""


def check_vtx_ids_column(vtx_ids_column: List[str]) -> None:
    """check argument to rise exception or log warning if needed."""
    if not all(isinstance(v, str) for v in vtx_ids_column):
        raise InvalidVtxIdArgument("All values in vtx_id column must be of type str")
