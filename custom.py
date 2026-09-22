from collections.abc import Callable

from numpy._core.numerictypes import float64
from numpy._typing import NDArray

# from typing import Any, Protocol

type IndependentVariable = NDArray[float64]
type DependentVariable = NDArray[float64]

type SampleFunc = Callable[[IndependentVariable], DependentVariable]
# class Interferogram(Protocol):
#     def __call__(
#         self,
#         independent: IndependentVariable,
#         *args: Any,
#         **kwargs: Any
#     ) -> DependentVariable:
#         ...

type FWHMLine = tuple[float, DependentVariable, DependentVariable] # (half_max, left_x, right_x)

# one FWHM contains all the lines and texts for a single interferogram
class FWHM:
    lines: list[FWHMLine]
    texts: list[str]

    def __init__(self, lines, texts):
        self.lines = lines  # Instance attribute
        self.texts = texts    # Instance attribute
