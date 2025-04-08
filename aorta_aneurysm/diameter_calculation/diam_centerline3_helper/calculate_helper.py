"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from typing import List

import SimpleITK as sitk
import numpy as np

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point
from aorta_aneurysm.util.array import cc
from aorta_aneurysm.util.image import get_xyz_array_from_image

def is_flipped_sanity_check(seg_iso_clean: sitk.Image) -> bool:
    try:
        _outer_arr = get_xyz_array_from_image( seg_iso_clean - sitk.BinaryErode(seg_iso_clean) )
        outer_points = [Point(x[0], x[1], x[2]) for x in np.argwhere(_outer_arr == 1)]
        max_z_point = max(outer_points, key=lambda p: p.z)
        min_z_point = min(outer_points, key=lambda p: p.z)
        max_y_point = max(outer_points, key=lambda p: p.y)
        min_y_point = min(outer_points, key=lambda p: p.y)
        max_z, min_z, max_y, min_y = max_z_point.z, min_z_point.z, max_y_point.y, min_y_point.y

        if max_z - min_z > 110:        
            left_side_points = [p for p in outer_points if p.y < max_z_point.y-5]
            right_side_points = [p for p in outer_points if p.y > min_z_point.y+5]

            min_z_left = min(left_side_points, key=lambda p: p.z).z
            max_z_left = max(left_side_points, key=lambda p: p.z).z
            min_z_right = min(right_side_points, key=lambda p: p.z).z
            max_z_right = max(right_side_points, key=lambda p: p.z).z

            if max_z_right -110 > max_z_left:
                return True
                    
    except Exception as e:
        pass
    return False