"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Vector import Vector

@dataclass
class Normal:
    """
    A class representing a normal vector in 3D space.
    """

    x: float
    y: float
    z: float

    _np_cache: np.array = field(default=None)
    _np_normalized_cache: np.array = field(default=None)

    def __post_init__(self):
        if abs(self.x) < 1e-5:
            self.x = 1e-5
        if abs(self.y) < 1e-5:
            self.y = 1e-5
        if abs(self.z) < 1e-5:
            self.z = 1e-5
        if self.x > 1.0 - 1e-5:
            self.x = 1.0-1e-5
        if self.y > 1.0 - 1e-5:
            self.y = 1.0-1e-5
        if self.z > 1.0 - 1e-5:
            self.z = 1.0-1e-5
        if self.x < -1.0 + 1e-5:
            self.x = -1.0+1e-5
        if self.y < -1.0 + 1e-5:
            self.y = -1.0+1e-5
        if self.z < -1.0 + 1e-5:
            self.z = -1.0+1e-5
        self._np_cache = np.array([self.x, self.y, self.z])
        self._np_normalized_cache = self._np_cache / np.linalg.norm(self._np_cache)
            

    def as_np(self):
        return self._np_cache
    
    def as_np_normalized(self):
        return self._np_normalized_cache

    def as_str(self):
        x, y, z = self.as_np_normalized().tolist()
        return f"Normal(norm: {x:.1f}, {y:.1f}, {z:.1f})"

    def is_ok(self):
        if (
            np.isnan(self.as_np()).any() or np.isinf(self.as_np()).any() or
            np.isnan(self.as_np_normalized()).any() or np.isinf(self.as_np_normalized()).any() or
            np.allclose(self.as_np(), np.zeros(3), atol=1e-6) or
            np.allclose(self.as_np_normalized(), np.zeros(3), atol=1e-6) or
            self.as_np_normalized()[0] > 1.0 or self.as_np_normalized()[0] < -1.0 or
            self.as_np_normalized()[1] > 1.0 or self.as_np_normalized()[1] < -1.0 or
            self.as_np_normalized()[2] > 1.0 or self.as_np_normalized()[2] < -1.0
        ):
            return False
        return True

def calculate_robust_normals2(centerline_points: List[Point], window_size: int = 3, do_by_xz_reversal_image_size: List[int] = None) -> List[Normal]:
    """
    Calculate a more robust normal vector at all indexes in the centerline points
    by averaging the cross products of vectors formed within a specified window size.
    
    Parameters
    ----------
    centerline_points (List[Point]): 
        List of centerline points.
    window_size (int): 
        Number of points on either side of the index to consider for averaging.
    
    Returns
    -------
    List[Normal]
        List of normal vectors.
    """

    if do_by_xz_reversal_image_size is not None:
        centerline_points = [Point(do_by_xz_reversal_image_size[0]-p.x, p.y, do_by_xz_reversal_image_size[2]-p.z) for p in centerline_points]

    normals = []
    # Iterate over all points in the centerline
    for index, point in enumerate(centerline_points):
        # Find all possible normals for point
        normals_p = []
        for i in range(index - window_size, index + window_size):
            i_prev = i - 1
            i_next = i + 1
            if i != index and i >= 0 and i < len(centerline_points):
                op = centerline_points[i]
                vectors = []
                if i_prev >= 0:
                    op_prev = centerline_points[i_prev]
                    if op.distance_to(op_prev) < 10:
                        vectors.append( Vector(p0 = op_prev, p1 = op) )
                if i_next < len(centerline_points):
                    op_next = centerline_points[i_next]
                    if op.distance_to(op_next) < 10:
                        vectors.append( Vector(p0 = op, p1 = op_next) )
                if len(vectors) > 0:
                    normal_np = np.average([v.as_np_normalized_oriented() for v in vectors], axis=0)
                    normal = Normal(x = normal_np[0], y = normal_np[1], z = normal_np[2])
                    if normal.is_ok():
                        normals_p.append(normal)

        # Average the normals
        mean_normal_np = np.mean([n.as_np_normalized() for n in normals_p], axis=0) if len(normals_p) else np.array([0, 0, 1])
        mean_normal = Normal(x = mean_normal_np[0], y = mean_normal_np[1], z = mean_normal_np[2])
        if mean_normal.is_ok():
            normals.append(mean_normal)
        else:
            normals.append(np.array([0, 0, 1])) # Fallback if no vectors are computed
    
    if do_by_xz_reversal_image_size is not None:
        normals = [Normal(-n.x, n.y, -n.z) for n in normals]

    return normals


def calculate_robust_normals(centerline_points: List[Point], window_size: int = 3) -> List[Normal]:
    """
    Calculate a more robust normal vector at all indexes in the centerline points
    by averaging the cross products of vectors formed within a specified window size.
    
    Parameters
    ----------
    centerline_points (List[Point]): 
        List of centerline points.
    window_size (int): 
        Number of points on either side of the index to consider for averaging.
    
    Returns
    -------
    List[Normal]
        List of normal vectors.
    """
    normals = []
    for index in range(len(centerline_points)):
        normal_vectors = []
        for i in range(index - window_size, index + window_size):
            i_prev = i - 1
            i_next = i + 1
            if i_prev >= 0 and i_next < len(centerline_points):
                p0 = centerline_points[i_prev].as_np()
                p1 = centerline_points[i].as_np()
                p2 = centerline_points[i_next].as_np()
                # v1 = p1 - p0
                # v2 = p2 - p1
                # normal = np.cross(v1, v2)
                normal = np.average([p1-p0, p2-p1], axis=0)
                # Check if NaN
                if not np.isnan(normal).any():
                    normal_vectors.append(normal)
            elif i_prev >= 0 and i < len(centerline_points):
                p0 = centerline_points[i_prev].as_np()
                p1 = centerline_points[i].as_np()
                normal = p1-p0
                # Check if NaN
                if not np.isnan(normal).any():
                    normal_vectors.append(normal)
        # Averaging the normals
        if len(normal_vectors):
            mean_normal = np.mean(normal_vectors, axis=0)
            if not np.isnan(mean_normal).any():
                normals.append(mean_normal)
            else:
                normals.append(np.array([0, 0, 0])) # Fallback if no vectors are computed
        else:
            normals.append(np.array([0, 0, 0])) # Fallback if no vectors are computed
    return [Normal(n[0], n[1], n[2]) for n in normals]

def normal_to_rotation_matrix(normal: Normal) -> np.ndarray:
    """Construct the Rotation Matrix Using Rodrigues' Rotation Formula"""
    n_normalized = normal.as_np_normalized()

    # Define the Z-axis vector
    k = np.array([0, 0, 1])

    # Compute the rotation axis (cross product of n_normalized and k)
    rotation_axis = np.cross(n_normalized, k)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize the rotation axis

    # Compute the angle of rotation (dot product of n_normalized and k)
    cos_theta = np.dot(n_normalized, k)
    theta = np.arccos(cos_theta)

    # Construct the skew-symmetric matrix for the rotation axis
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])

    # Rodrigues' rotation formula to find the rotation matrix
    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (np.dot(K, K))

    return R

