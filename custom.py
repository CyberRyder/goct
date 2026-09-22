from collections.abc import Callable

from numpy._core.numerictypes import float64
from numpy._typing import NDArray

type IndependentVariable = NDArray[float64]
type DependentVariable = NDArray[float64]

type SampleFunc = Callable[[IndependentVariable], IndependentVariable]
