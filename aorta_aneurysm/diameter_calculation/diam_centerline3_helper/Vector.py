"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from dataclasses import dataclass, field

import numpy as np

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point

@dataclass
class Vector:
    p0: Point
    p1: Point

    _np_cache: np.array = field(default=None)
    _np_normalized_cache: np.array = field(default=None)
    _np_normalized_oriented_cache: np.array = field(default=None)

    def as_np(self):
        if self._np_cache is None:
            self._np_cache = self.p1.as_np() - self.p0.as_np()
        return self._np_cache

    def as_np_normalized(self):
        if self._np_normalized_cache is None:
            self._np_normalized_cache = self.as_np() / np.linalg.norm(self.as_np())
        return self._np_normalized_cache
    
    def as_np_normalized_oriented(self):
        if self._np_normalized_oriented_cache is None:
            vec_np = self.as_np_normalized()
            x, y, z = tuple(vec_np.tolist())
            if abs(z) > 0.1:
                self._np_normalized_oriented_cache = vec_np if z>0 else -vec_np
            elif abs(y) > 0.1:
                self._np_normalized_oriented_cache = vec_np if y>0 else -vec_np
            elif abs(x) > 0.1:
                self._np_normalized_oriented_cache = vec_np if x>0 else -vec_np
            else:
                self._np_normalized_oriented_cache = vec_np
        return self._np_normalized_oriented_cache