"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from dataclasses import dataclass, field

from matplotlib.axes import Axes

@dataclass
class DiameterCalculationResult:
    """Dataclass to store the result of the diameter calculation."""
    diameter: float = -1
    slice_num: int = -1
    
    calculation_success: bool = True
    error: str = "-"
    
    diameter_asc : float = -1
    slice_num_asc : int = -1
    
    diameter_dsc : float = -1
    slice_num_dsc : int = -1

    diameter_arch : float = -1
    slice_num_arch : int = -1

    plot: Axes = None
    
    other_info: dict = field(default_factory=dict)


    def __post_init__(self):
        self.other_info = dict() if self.other_info is None else self.other_info