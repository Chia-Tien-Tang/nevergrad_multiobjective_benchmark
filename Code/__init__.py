# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides a collection of multi-objective optimization functions.
It includes various test functions commonly used in multi-objective optimization research.
"""

# Importing multi-objective optimization functions and core classes
from .functions import (
    BinhKorn, ChankongHaimes, PoloniTwoObjective, TestFunction4,
    CTP1, ConstrEx, FonsecaFleming, Kursawe, ZDT1, ZDT2, ZDT3,
    ZDT4, ZDT6, OsyczkaKundu, Viennet, Schaffer1, Schaffer2
)
from .core import ConstrainedMultiObjective

from nevergrad.common import errors
import nevergrad.common.typing as tp


# This class is maintained for backward compatibility but is no longer recommended for use.
class MultiobjectiveFunction:
    """MultiobjectiveFunction is deprecated and is removed after v0.4.3 "
    because it is no more needed. You should just pass a multiobjective loss to "
    optimizer.tell.\nSee https://facebookresearch.github.io/nevergrad/"
    optimization.html#multiobjective-minimization-with-nevergrad\n",
    """

    def __init__(
        self,
        multiobjective_function: tp.Callable[..., tp.ArrayLike],
        upper_bounds: tp.Optional[tp.ArrayLike] = None,
    ) -> None:
        raise errors.NevergradDeprecationError(
            "MultiobjectiveFunction is deprecated and is removed after v0.4.3 "
            "because it is no more needed. You should just pass a multiobjective loss to "
            "optimizer.tell.\nSee https://facebookresearch.github.io/nevergrad/"
            "optimization.html#multiobjective-minimization-with-nevergrad\n",
        )
