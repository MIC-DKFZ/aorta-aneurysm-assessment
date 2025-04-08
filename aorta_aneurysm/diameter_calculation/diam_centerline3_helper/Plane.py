"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Normal import Normal, normal_to_rotation_matrix
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point, rotate_point


@dataclass
class Plane:
    """
    A class representing a plane in 3D space.
    
    The plane is defined by a point and a normal vector. The plane can be used to check if points lie on the plane.
    """

    point: Point
    normal: Normal
    voxel_size: float = 1.0

    def as_str(self):
        pl_str = self.point.as_str()
        no_str = self.normal.as_str()
        return f"Plane[ \t {pl_str} \t {no_str} \t ]"

    def is_on_plane(self, point: Point, tolerance: float = 1.0) -> bool:
        """
        Check if given point lies on the plane.
        
        Parameters
        ----------
        point: Point
            The point to check.
        
        Returns
        -------
        bool
            Whether the point lies on the plane.
        """
        point_vector = point.as_np() - self.point.as_np()
        dot_product = np.abs(np.dot(point_vector, self.normal.as_np_normalized()))
        return dot_product < tolerance
    
    def calculate_diameter(self, points: List[Point]) -> float:
        """
        Calculate the pairwise distances between all points in the list of points.

        Parameters
        ----------
        points: List[Point]
            The list of points in the plane.
        """
        raise ValueError("Plane.calculate_diameter is deprecated.")


def find_points_on_plane(image_shape_3D: Tuple[int, int, int], plane: Plane, tolerance: float = 1e-3) -> np.ndarray:
    """
    Determine which points in a 3D image are on a plane.

    Parameters
    ----------
    image_shape_3D: Tuple[int, int, int]
        The shape of the 3D image.
    plane: Plane
        The plane to check.
    tolerance: float, optional
        The tolerance for the dot product of the point vector and the normal vector to be considered on the plane.
    
    Returns
    -------
    np.ndarray
        A boolean array (same shape as the input provided) indicating which points are on the plane.
    """
    X, Y, Z = image_shape_3D

    # Generate grid of points
    x = np.arange(X)
    y = np.arange(Y)
    z = np.arange(Z)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    
    # Convert plane point and normal to numpy arrays
    pp_np = plane.point.as_np()
    pn_np = plane.normal.as_np_normalized()
    
    # Compute the plane equation for each point
    points = np.stack((grid_x, grid_y, grid_z), axis=-1)  # Shape will be (X, Y, Z, 3)
    point_vectors = points - pp_np
    dot_products = np.abs(np.einsum('ijkl,l->ijk', point_vectors, pn_np))  # Efficiently compute dot product
    
    # Determine which points are on the plane
    on_plane = dot_products < tolerance
    
    return on_plane


def find_points_on_plane_with_seg(
        seg_arr: np.ndarray,
        plane: Plane, 
        tolerance: float = 0.5 # 1e-3,
        ) -> np.ndarray:
    """
    Determine which points in a 3D seg are on a plane, hopefully faster

    Parameters
    ----------
    seg_arr: np.ndarray
        The segmentation array to check if the points are on the segmentation.
    plane: Plane
        The plane to check.
    tolerance: float, optional
        The tolerance for the dot product of the point vector and the normal vector to be considered on the plane.
    
    Returns
    -------
    np.ndarray
        A boolean array (same shape as the input provided) indicating which points are on the plane.
    """
    # Plane properties
    pp_np = plane.point.as_np()
    pn_np = plane.normal.as_np_normalized()

    # Indices of non-zero (or interesting) points in the segmentation
    non_zero_indices = np.nonzero(seg_arr)
    points = np.array(non_zero_indices).T  # Transpose to get points in (N, 3) shape

    # Calculate vector from plane point to each non-zero point
    point_vectors = points - pp_np

    # Dot product and check if points are on the plane
    dot_products = np.abs(np.einsum('ij,j->i', point_vectors, pn_np))
    on_plane_indices = non_zero_indices[0][dot_products < tolerance], non_zero_indices[1][dot_products < tolerance], non_zero_indices[2][dot_products < tolerance]

    # Create a result array (same shape as segmentation)
    result = np.zeros(seg_arr.shape, dtype=bool)
    result[on_plane_indices] = True  # Set True for indices where points are on the plane

    return result




def slice_volume_with_plane(volume_arr: np.ndarray, plane: Plane, 
                            return_rotation_and_translation: bool = False,
                            is_seg: bool = False
                            ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Slice a 3D volume with a plane.
    
    Parameters
    ----------
    volume_arr: np.ndarray
        The 3D volume to slice.
    plane: Plane
        The plane to slice with.
    return_rotation_and_translation: bool, optional
        Whether to return the rotation matrix and translation applied to the slice, by default False.
    is_seg: bool, optional
        Whether the volume is a segmentation, by default False. Seg can be used without this flag, but it will be slower.
        
    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int]]]
        The 2D slice of the volume or the 2D slice of the volume and the translation applied to the slice.
    """
    volume_3D_shape = volume_arr.shape

    R = normal_to_rotation_matrix(plane.normal)
    ## Find which points are on the plane
    if not is_seg:
        points_mask = find_points_on_plane(volume_3D_shape, plane, tolerance=1.0)
    else:
        points_mask = find_points_on_plane_with_seg(volume_arr, plane, tolerance=1.0)
    ## Collect the relevant points
    points_vol = [
        Point(x,y,z, value=volume_arr[x,y,z]) for x,y,z in zip(*np.where(points_mask)) if volume_arr[x,y,z] > 0
    ]
    ## Safety check
    if not len(points_vol):
        return np.zeros(volume_3D_shape[0:2]) if not return_rotation_and_translation else (np.zeros(volume_3D_shape[0:2]), R, np.array([0, 0]))
    ## Rotate the points
    rotated_points_vol = [rotate_point(p, R) for p in points_vol]
    ## Drop third dim and get them as np indices
    out_indices = np.array([(p.x, p.y) for p in rotated_points_vol])
    ## Translate to positive indices
    min_indices = np.min(out_indices, axis=0)
    if np.any(min_indices < 0):
        translation = -min_indices
        out_indices = out_indices - min_indices
        rotated_points_vol = [Point(p.x - min_indices[0], p.y - min_indices[1], p.z, value=p.value) for p in rotated_points_vol]
    else:
        translation = np.array([0, 0])
    ## Remove indices larger than target shape
    values_to_transfer_vol = [rp.value for rp in rotated_points_vol if rp.as_np()[0] < volume_3D_shape[0] and rp.as_np()[1] < volume_3D_shape[1]]
    out_indices = out_indices[out_indices[:,0] < volume_3D_shape[0], :]
    out_indices = out_indices[out_indices[:,1] < volume_3D_shape[1], :]
    ## Construct 2D output
    slice_arr = np.zeros(volume_3D_shape[0:2])
    ## Fill the 2D output
    slice_arr[out_indices[:,0], out_indices[:,1]] = values_to_transfer_vol
    ## Finish
    return slice_arr if not return_rotation_and_translation else (slice_arr, R, translation)


def slice_image_and_seg_with_plane(
        img_arr: np.ndarray, seg_arr: np.ndarray, plane: Plane, tolerance: float = 1.0,
        return_rotation_and_translation: bool = False
    ) -> Union[ Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] ]:
    """
    Slice a 3D image and segmentation with a plane.

    Parameters
    ----------
    img_arr: np.ndarray
        The 3D image to slice.
    seg_arr: np.ndarray
        The 3D segmentation to slice.
    plane: Plane
        The plane to slice with.
    tolerance: float, optional
        The tolerance for the dot product of the point vector and the normal vector to be considered on the plane.
    return_rotation_and_translation: bool, optional
        Whether to return the rotation matrix and translation applied to the slice, by default False.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The 2D slices of the image and segmentation.
    """
    volume_3D_shape = seg_arr.shape

    R = normal_to_rotation_matrix(plane.normal)
    ## Find which points are on the plane
    points_mask = find_points_on_plane(volume_3D_shape, plane, tolerance=tolerance)
    ## Collect the relevant points
    points_img = [
        Point(x,y,z, value=img_arr[x,y,z]) for x,y,z in zip(*np.where(points_mask))
    ]
    points_seg = [
        Point(x,y,z, value=seg_arr[x,y,z]) for x,y,z in zip(*np.where(points_mask)) if seg_arr[x,y,z] > 0
    ]
    ## Rotate the points
    rotated_points_img = [rotate_point(p, R) for p in points_img]
    rotated_points_seg = [rotate_point(p, R) for p in points_seg]
    ## Drop third dim and get them as np indices
    out_indices_img = np.array([(p.x, p.y) for p in rotated_points_img])
    out_indices_seg = np.array([(p.x, p.y) for p in rotated_points_seg])
    ## Translate to positive indices
    min_indices = np.min(out_indices_img, axis=0)
    if np.any(min_indices < 0):
        translation = -min_indices
        out_indices_img = out_indices_img - min_indices
        out_indices_seg = out_indices_seg - min_indices
        rotated_points_img = [Point(p.x - min_indices[0], p.y - min_indices[1], p.z, value=p.value) for p in rotated_points_img]
        rotated_points_seg = [Point(p.x - min_indices[0], p.y - min_indices[1], p.z, value=p.value) for p in rotated_points_seg]
    else:
        translation = np.array([0, 0])
    ## Remove indices larger than target shape
    values_to_transfer_img = [rp.value for rp in rotated_points_img if rp.as_np()[0] < volume_3D_shape[0] and rp.as_np()[1] < volume_3D_shape[1]]
    values_to_transfer_seg = [rp.value for rp in rotated_points_seg if rp.as_np()[0] < volume_3D_shape[0] and rp.as_np()[1] < volume_3D_shape[1]]
    out_indices_img = out_indices_img[out_indices_img[:,0] < volume_3D_shape[0], :]
    out_indices_img = out_indices_img[out_indices_img[:,1] < volume_3D_shape[1], :]
    out_indices_seg = out_indices_seg[out_indices_seg[:,0] < volume_3D_shape[0], :]
    out_indices_seg = out_indices_seg[out_indices_seg[:,1] < volume_3D_shape[1], :]
    ## Construct 2D output
    slice_arr_img = np.zeros(volume_3D_shape[0:2])
    slice_arr_seg = np.zeros(volume_3D_shape[0:2])
    ## Fill the 2D output
    slice_arr_img[out_indices_img[:,0], out_indices_img[:,1]] = values_to_transfer_img
    slice_arr_seg[out_indices_seg[:,0], out_indices_seg[:,1]] = values_to_transfer_seg
    ## Finish
    return (slice_arr_img, slice_arr_seg) if not return_rotation_and_translation else (slice_arr_img, slice_arr_seg, R, translation)

