"""
This file contains the config class and a validator that ensures the
experimental configuration is physically valid and will not cause errors.
"""

from pathlib import Path
from typing import Literal

import tomllib
from pydantic import BaseModel, Field, model_validator


class GraphingConfig(BaseModel):
    oct_type: Literal["monochromatic", "nonmonochromatic", "quantum"]
    grover: bool
    time_limit: float = Field(gt=0)  # seconds
    y_limit: float = Field(gt=0)  # unitless
    display_interferogram: bool
    display_scaled_interferogram: bool
    scale_max: bool
    display_sample: bool


class SampleConfig(BaseModel):
    length: float = Field(gt=0)  # meters
    refractive_index: float = Field(gt=1.0)  # unitless, >1 physically
    initial_depth: float = Field(gt=0)  # micrometers
    entering_reflectance: float = Field(ge=0, le=1)
    entering_reflectance_narrowness: float = Field(gt=0)
    exiting_reflectance: float = Field(ge=0, le=1)
    exiting_reflectance_narrowness: float = Field(gt=0)
    shape: Literal["gaussian", "cornered", "square"]


class LaserConfig(BaseModel):
    central_wavelength: float = Field(gt=0)  # micrometers
    spectral_width: float = Field(gt=0)  # radians / second


class ExperimentConfig(BaseModel):
    graphing: GraphingConfig
    sample: SampleConfig
    laser: LaserConfig

    @model_validator(mode="after")
    def check_cross_field_constraints(self) -> "ExperimentConfig":
        # validates anything that depends on more than one section
        # currently blank but could fill in (e.g. that the entire sample fits within the time limit)
        return self

    @classmethod
    def from_toml(cls, path: Path | str) -> "ExperimentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with open(path, "rb") as f:
            try:
                data = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise ValueError(f"invalid TOML in {path}: {e}") from e
        return cls(**data)


def load_config() -> ExperimentConfig:
    return ExperimentConfig.from_toml("config.toml")
