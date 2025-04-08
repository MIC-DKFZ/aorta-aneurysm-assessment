"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import random, copy
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from typing import List, Dict, Tuple, Any, Union

import numpy as np

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Normal import Normal, calculate_robust_normals, calculate_robust_normals2
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point, order_centerline_points, remove_close_centerline_points, rotate_point, centerline_dijkstra
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Plane import Plane, slice_volume_with_plane

def __categorize_part_of_arch_at_start(point: Point, normal: Normal, all_other_points: List[Point]) -> List[str]:
    point_arr = np.array([point.x, point.y, point.z])
    all_other_points_arr = np.array([[point.x, point.y, point.z] for point in all_other_points])
    xyz_signed_distances_arr = all_other_points_arr - point_arr
    max_z_distance = np.max(xyz_signed_distances_arr[:,2])
    if max_z_distance < 30 and normal.is_ok() and normal.as_np_normalized()[1] < -0.50 and normal.as_np_normalized()[2] < 0.7:
        return ['ARCH']
    else:
        return ['ASC', 'DSC', 'ARCH']

def __categorize_aorta_plane_location_based_on_point(point: Point, all_other_points: List[Point]) -> List[str]:
    """Categorize aorta point by location
    
    Parameters
    ----------
    point : Point
        Point to categorize
    all_other_points : List[Point]
        Other points to compare with
    
    Returns
    -------
    List[str]
        Possible categories of the point, including: 'DSC', 'ARCH', 'ASC'
    """
    point_arr = np.array([point.x, point.y, point.z])
    all_other_points_arr = np.array([[point.x, point.y, point.z] for point in all_other_points])

    xyz_signed_distances_arr = all_other_points_arr - point_arr
    
    max_x_distance = np.max(xyz_signed_distances_arr[:,0])
    min_x_distance = np.min(xyz_signed_distances_arr[:,0])
    max_y_distance = np.max(xyz_signed_distances_arr[:,1])
    min_y_distance = np.min(xyz_signed_distances_arr[:,1])
    max_z_distance = np.max(xyz_signed_distances_arr[:,2])
    min_z_distance = np.min(xyz_signed_distances_arr[:,2])

    if max_z_distance > 120:
        return ['DSC']

    options = ['ASC', 'DSC', 'ARCH']

    # z
    if max_z_distance > 45:
        options = [o for o in options if o != 'ARCH']
    elif max_z_distance > 60:
        options = [o for o in options if o != 'ARCH']
        options = [o for o in options if o != 'ASC']

    # y
    if max_y_distance > 30:
        options = [o for o in options if o != 'DSC']
    elif min_y_distance < -30:
        options = [o for o in options if o != 'ASC']

    # arch
    if max_z_distance < 10 and min_y_distance < -25 and max_y_distance > 25:
        options = [o for o in options if o != 'ASC']
        options = [o for o in options if o != 'DSC']

    return options

def __categorize_aorta_plane_location_based_on_general_image(point, normal, voxel_size, image_shape):
    x, y, z = point.as_np()
    nx, ny, nz = normal.as_np_normalized()

    upper_side = point.z > (image_shape[2] // 4)

    if not upper_side:
        return ['DSC']
    
    if ny < -0.35:
        return ['ASC', 'DSC', 'ARCH'] # could have been DSC, ARCH only, but sometimes ASC is weird at the start

    elif ny > 0.4:
        return ['ASC', 'ARCH']

    ## Uncertain
    return ['ASC', 'DSC', 'ARCH']

def __categorize_aorta_plane_location_based_on_other_points(point, current_categories, all_other_points, all_other_categories, lenient=False):
    point_arr = np.array([point.x, point.y, point.z])
    all_other_points_arr = np.array([[point.x, point.y, point.z] for point in all_other_points])
    xyz_signed_distances_arr = all_other_points_arr - point_arr

    if len(current_categories) == 1:
        return current_categories

    THRES_XY_ASC_DSC = 8 if not lenient else 12
    THRES_Z_ASC_DSC = 10 if not lenient else 20
    THRES_XZ_ARCH = 3 if not lenient else 3
    THRES_Y_ARCH = 2 if not lenient else 2

    for o_point, o_cats, od in zip(all_other_points, all_other_categories, xyz_signed_distances_arr):
        od_x, od_y, od_z = od

        if len(o_cats) == 1: # if the other is decided
            o_cat = o_cats[0]
            if o_cat != 'ARCH': # if arch is not possible
                if abs(od_x)+abs(od_y) <= THRES_XY_ASC_DSC and abs(od_z) <= THRES_Z_ASC_DSC: # roughly on the same-ish line and close
                    if o_cat in current_categories:
                        return o_cats
            if o_cat == 'ARCH' and 'ARCH' in current_categories:
                if abs(od_x)+abs(od_z) <= THRES_XZ_ARCH and abs(od_y) <= THRES_Y_ARCH: # roughly on the same line, pretty strict
                    return o_cats
                if od_z < 0: # higher than known arch
                    return o_cats

    return current_categories

def __categorize_aorta_plane_location_based_on_normal(normal: Normal) -> List[str]:
    """Categorize aorta location based on normal
    
    Parameters
    ----------
    normal : Normal
        Normal to categorize
    
    Returns
    -------
    List[str]
        Possible categories of the point, including: 'DSC', 'ARCH', 'ASC'
    """
    n = [round(x,2) for x in normal.as_np_normalized()]

    x, y, z = n[0], n[1], n[2]

    x_pos, y_pos, z_pos = abs(x), abs(y), abs(z)

    if z_pos < 0.4:
        return ['ARCH']
    elif z_pos < 0.7: # ARCH still possible
        if y > 0.3:
            return ['ASC', 'ARCH']
        elif y < -0.3:
            return ['DSC', 'ARCH']
        else:
            return ['ASC', 'DSC', 'ARCH']

    else: # ARCH not possible
        if y > 0.35:
            return ['ASC']
        elif y < -0.35:
            return ['DSC']
        else:
            return ['ASC', 'DSC']


def categorize_planes(planes: List[Plane], voxel_size: float, image_shape: Tuple) -> List[str]:
    """Categorize aorta plane
    
    Parameters
    ----------
    planes: List[Plane]
        List of planes to categorize
    voxel_size: float
        Voxel size
    
    Returns
    -------
    str
        Category of each plane, including: 'DSC', 'ARCH', 'ASC'
    """
    ## Handle single plane case
    if len(planes) == 1:
        cats = __categorize_aorta_plane_location_based_on_normal(planes[0].normal)
        if len(cats) == 1:
            return cats
        return ['ASC']

    planes_possible_categories = []

    ## Just generally based on location relative to other points
    for idx in range(len(planes)):
        point = planes[idx].point
        other_points = [pl.point for pl in planes if pl.point != point]
        categories_loc = __categorize_aorta_plane_location_based_on_point(point, other_points)
        planes_possible_categories.append(categories_loc)

    ## Based on other points classification (1st time)
    for idx in range(len(planes)):
        point = planes[idx].point
        cats = planes_possible_categories[idx]
        if len(cats) > 1:
            other_points = [pl.point for pl in planes if pl.point != point]
            other_categorizations = []
            for idx2 in range(len(planes)):
                if idx!=idx2:
                    other_categorizations.append(planes_possible_categories[idx2])
            planes_possible_categories[idx] = __categorize_aorta_plane_location_based_on_other_points(point, cats, other_points, other_categorizations)
            other_categorizations = []
            for idx2 in range(len(planes)):
                if idx!=idx2:
                    other_categorizations.append(planes_possible_categories[idx2])
            planes_possible_categories[idx] = __categorize_aorta_plane_location_based_on_other_points(point, cats, other_points, other_categorizations)

    for idx in range(len(planes)):
        plane = planes[idx]
        categories_gen = __categorize_aorta_plane_location_based_on_general_image(plane.point, plane.normal, voxel_size, image_shape)
        new_cats = []
        for cat in categories_gen:
            if cat in planes_possible_categories[idx]:
                new_cats.append(cat)
        if len(new_cats) > 0:
            planes_possible_categories[idx] = new_cats # otherwise keep the old ones

    ## Based on other points classification
    for idx in range(len(planes)):
        point = planes[idx].point
        cats = planes_possible_categories[idx]
        if len(cats) > 1:
            other_points = [pl.point for pl in planes if pl.point != point]
            other_categorizations = []
            for idx2 in range(len(planes)):
                if idx!=idx2:
                    other_categorizations.append(planes_possible_categories[idx2])
            planes_possible_categories[idx] = __categorize_aorta_plane_location_based_on_other_points(point, cats, other_points, other_categorizations)
            other_categorizations = []
            for idx2 in range(len(planes)):
                if idx!=idx2:
                    other_categorizations.append(planes_possible_categories[idx2])
            planes_possible_categories[idx] = __categorize_aorta_plane_location_based_on_other_points(point, cats, other_points, other_categorizations)

    ## Based on normals but a bit more careful
    for idx in range(len(planes)):
        normal = planes[idx].normal
        categories_nor = __categorize_aorta_plane_location_based_on_normal(normal)
        # if len(categories_nor) == 1 and categories_nor[0] in planes_possible_categories[idx]:
        #     planes_possible_categories[idx] = categories_nor
        new_cats = []
        for cat in categories_nor:
            if cat in planes_possible_categories[idx]:
                new_cats.append(cat)
        if len(new_cats) > 0:
            planes_possible_categories[idx] = new_cats # otherwise keep the old ones

    ## Based on other points classification
    for idx in range(len(planes)):
        point = planes[idx].point
        cats = planes_possible_categories[idx]
        if len(cats) > 1:
            other_points = [pl.point for pl in planes if pl.point != point]
            other_categorizations = []
            for idx2 in range(len(planes)):
                if idx!=idx2:
                    other_categorizations.append(planes_possible_categories[idx2])
            planes_possible_categories[idx] = __categorize_aorta_plane_location_based_on_other_points(point, cats, other_points, other_categorizations, lenient=True)
            other_categorizations = []
            for idx2 in range(len(planes)):
                if idx!=idx2:
                    other_categorizations.append(planes_possible_categories[idx2])
            planes_possible_categories[idx] = __categorize_aorta_plane_location_based_on_other_points(point, cats, other_points, other_categorizations, lenient=True)

    ## (Almost) final decision
    planes_categories = ['ASC'] * len(planes)
    for idx in range(len(planes)):
        cats = planes_possible_categories[idx]
        if len(cats) == 0:
            planes_categories[idx] = 'ASC'
        elif len(cats) == 1:
            planes_categories[idx] = cats[0]
        elif len(cats) == 2:
            if 'ASC' in cats:
                planes_categories[idx] = 'ASC'
            else:
                planes_categories[idx] = 'DSC'
        else: # 3
            planes_categories[idx] = 'ASC'
    
    ## Check for obvious error
    if 'ASC' in planes_categories and 'DSC' in planes_categories:
        min_y = min(plane.point.y for plane in planes)
        max_y = max(plane.point.y for plane in planes)
        if max_y - min_y <= 40: # all on roughly the same line
            min_z = min(plane.point.z for plane in planes)
            max_z = max(plane.point.z for plane in planes)
            # If tilting right and centerline point close to the edge (basically hitting the edge)
            y_of_point_with_min_z = [plane.point.y for plane in planes if plane.point.z == min_z][0]
            y_of_point_with_max_z = [plane.point.y for plane in planes if plane.point.z == max_z][0]
            if (
                y_of_point_with_max_z > y_of_point_with_min_z and 
                (image_shape[2] - y_of_point_with_max_z)*voxel_size < 15
            ):
                planes_categories = ['ASC'] * len(planes)
            else:
                planes_categories = ['DSC'] * len(planes)

    return planes_categories
