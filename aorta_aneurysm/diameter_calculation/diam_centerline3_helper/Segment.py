"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from tqdm import tqdm

import random, copy
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Union

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.morphology import skeletonize
from scipy.ndimage import binary_erosion

from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Normal import Normal, calculate_robust_normals, calculate_robust_normals2
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Point import Point, order_centerline_points, remove_close_centerline_points, rotate_point, centerline_dijkstra
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.Plane import Plane, slice_volume_with_plane
from aorta_aneurysm.diameter_calculation.diam_centerline3_helper.calc_centerline import get_aorta_centerline
from aorta_aneurysm.util.image import image_resample_to_spacing, get_xyz_array_from_image, get_image_from_xyz_array, image_keep_at_most_2_axial_cc
from aorta_aneurysm.util.array import keep_at_most_2_axial_cc, cc, calc_circularity, erode_only_in_xy

@dataclass
class Segment:
    """
    A class representing a segment (connected part of a segmentation) in 3D space.
    """

    id: int
    centerline_points: List[Point]
    outer_points: List[Point]
    voxel_size: float = 0.85
    window_size: int = 2
    keep_only_planes_with_circular_seg: bool = True
    seg_s_arr_only_for_keep_only_planes_with_circular_seg: np.ndarray = None # will be deleted after post init

    planes: List[Plane] = field(default=list) # automatically generated
    planes_points: List[List[Point]] = field(default_factory=list) # automatically generated
    planes_diameters: List[float] = field(default_factory=list) # automatically generated
    planes_categorization: List[str] = field(default_factory=list) # populated by create_segments

    def __post_init__(self):
        # centerline_normals = calculate_robust_normals(self.centerline_points, window_size=self.window_size)
        centerline_normals = calculate_robust_normals2(self.centerline_points, window_size=self.window_size)
        ## Get the planes
        self.planes = []
        idxs_to_remove = []
        for p,n in zip(self.centerline_points, centerline_normals):
            if (
                not np.isnan(n.as_np_normalized()).any() and not np.isnan(n.as_np()).any() and
                not n.as_np().tolist() == [0, 0, 0] and not n.as_np().tolist() == [0.0, 0.0, 0.0]
            ):
                plane = Plane(p, n, self.voxel_size)
                if not self.keep_only_planes_with_circular_seg:
                    self.planes.append(plane)
                else:
                    slice_arr, R, translation = slice_volume_with_plane(self.seg_s_arr_only_for_keep_only_planes_with_circular_seg, 
                                                                        plane, return_rotation_and_translation=True)
                    new_plane_point_2D_np = rotate_point(plane.point, R).as_np()[0:2] + translation
                    slice_arrs = cc(slice_arr, threshold_to_keep=50) # thres arbitrary, just in case
                    max_circularity_found = 0
                    for arr in slice_arrs:
                        try:
                            min_distance = np.min(np.linalg.norm(np.argwhere(arr == 1) - new_plane_point_2D_np, axis=1))
                            if min_distance < 10:
                                circularity = calc_circularity(arr, destructive=True)
                                max_circularity_found = max(max_circularity_found, circularity)
                            else:
                                # print(f"\tNote: min_distance={round(min_distance)} (len(slice_arrs)={len(slice_arrs)})")
                                pass
                        except Exception as e:
                            print(f"Error: {e} -- skipping")
                    if round(max_circularity_found,2) < 0.85 - 1e-6:
                        # print(f"Not keeping centerline point (max_circularity_found={round(max_circularity_found,2)}) (len(slice_arrs)={len(slice_arrs)})")
                        idxs_to_remove.append(self.centerline_points.index(p))
                    else:
                        self.planes.append(plane)
        ## Free this memory, because this is not needed anymore if it was provided
        if self.seg_s_arr_only_for_keep_only_planes_with_circular_seg is not None:
            del self.seg_s_arr_only_for_keep_only_planes_with_circular_seg
        ## Remove the points that were not circular enough or had a problem with the plane cc'ing
        for idx in idxs_to_remove[::-1]:
            del self.centerline_points[idx]
        ## Get the plane points
        self.planes_points = []
        for plane in self.planes:
            _is_on_plane = [plane.is_on_plane(p) for p in self.outer_points]
            _points_to_keep = [p for p, is_on_plane in zip(self.outer_points, _is_on_plane) if is_on_plane]
            self.planes_points.append(_points_to_keep)
        ## Get the plane diameters
        self.planes_diameters = [plane.calculate_diameter(points) for plane, points in zip(self.planes, self.planes_points)]


def get_max_asc_dsc_arch_planes(segments: List[Segment]) -> Tuple[Union[Plane, None], float, Union[Plane, None], float, Union[Plane, None], float]:
    """
    Get the maximum planes for the ASC, DSC, and ARCH categories.

    Parameters
    ----------
    segments: List[Segment]
        The list of segments.
    
    Returns
    -------
    Tuple[Union[Plane, None], float, Union[Plane, None], float, Union[Plane, None], float]
        Max asc plane, max asc diameter, max dsc plane, max dsc diameter, max arch plane, max arch diameter
    """
    category_to_plane_and_diam : Dict[str, Tuple[Plane,float]] = {'ASC': [], 'DSC': [], 'ARCH': []}
    for segment in segments:
        for plane,diam,cat in zip(segment.planes, segment.planes_diameters, segment.planes_categorization):
            category_to_plane_and_diam[cat].append((plane,diam))
    planes_collection : Dict[str, Union[Plane, None]] = {'ASC': None, 'DSC': None, 'ARCH': None}
    diams_collection : Dict[str, float] = {'ASC': 0, 'DSC': 0, 'ARCH': 0}
    for cat in category_to_plane_and_diam:
        if len(category_to_plane_and_diam[cat]) > 0:
            # Sort by diam
            category_to_plane_and_diam[cat].sort(key=lambda x: x[1], reverse=True)
            planes_collection[cat] = category_to_plane_and_diam[cat][0][0]
            diams_collection[cat] = category_to_plane_and_diam[cat][0][1]
    return (
        planes_collection['ASC'], diams_collection['ASC'], 
        planes_collection['DSC'], diams_collection['DSC'],
        planes_collection['ARCH'], diams_collection['ARCH']
    )

def create_segments(
        seg: sitk.Image, 
        override_min_spacing_used: float = None,
        centerline_calculation_erode_iterations: int = 2,
        volume_threshold_keep_connected_components: int = 1000,
        outer_points_to_keep: int = 10000,
        order_centerline_points_distance_threshold: float = 2.0,
        remove_close_centerline_points_distance_threshold: float = 2.0,
        keep_only_planes_with_circular_seg: bool = True,
        verbose: bool = False
    ) -> List[Segment]:
    """
    Create segments from a segmentation.
    
    Parameters
    ----------
    seg: sitk.Image
        The segmentation image.
    override_min_spacing_used: float, optional
        The minimum spacing used for resampling.
    centerline_calculation_erode_iterations: int, optional
        The number of iterations for the centerline calculation.
    volume_threshold_keep_connected_components: int, optional
        The volume threshold to keep connected components.
    outer_points_to_keep: int, optional
        The number of outer points to keep. Controls speed - 5000 takes around 10 seconds. 1000 is very fast.
    order_centerline_points_distance_threshold: float, optional
        The distance threshold for ordering centerline points.
    remove_close_centerline_points_distance_threshold: float, optional
        The distance threshold for removing close centerline points.
    keep_only_planes_with_circular_seg: bool, optional
        Whether to keep only planes with circular segments. Slow but effective.
    verbose: bool, optional
        Whether to print debug information.
        
    Returns
    -------
    List[Segment]
        The list of segments.
    """
    ## Resample
    print('create_segments: Resampling...') if verbose else None
    min_sp = min(seg.GetSpacing()) if override_min_spacing_used is None else override_min_spacing_used
    seg = image_resample_to_spacing(seg, (min_sp, min_sp, min_sp), is_label=True)
    
    ## Keep at most 2 connected components    
    print('create_segments: Keeping at most 2 connected components...') if verbose else None
    seg = image_keep_at_most_2_axial_cc(seg)
    
    ## Create centerline
    print('create_segments: Creating centerline...') if verbose else None
    seg_arr = get_xyz_array_from_image(seg)
    # _eroded = binary_erosion(seg_arr, iterations=centerline_calculation_erode_iterations)
    # centerline_arr = skeletonize(_eroded)
    _, centerline_arr = get_aorta_centerline(seg_arr, min_sp)
    
    ## Get the outer volume points
    print('create_segments: Getting outer volume points...') if verbose else None
    outer_arr = get_xyz_array_from_image( seg - sitk.BinaryErode(seg) )
    
    ## Split
    print('create_segments: Splitting...') if verbose else None
    seg_b_arrs = cc(seg_arr, threshold_to_keep=volume_threshold_keep_connected_components)
    centerline_b_arrs = []
    outer_b_arrs = []
    for id_segment, seg_binary_arr in tqdm(enumerate(seg_b_arrs), desc='create_segments: Seg->Points', disable=not verbose):
        centerline_b_arrs.append( np.logical_and(centerline_arr, seg_binary_arr) )
        outer_b_arrs.append( np.logical_and(outer_arr, seg_binary_arr) )
    
    ## Creating segments
    print('create_segments: Creating segments...') if verbose else None
    segments = []
    number_of_segments = len(seg_b_arrs)
    for id_segment, (seg_s_arr, centerline_s_arr, outer_s_arr) in enumerate(zip(
        seg_b_arrs, centerline_b_arrs, outer_b_arrs
    )):
        ## Reduce outer points
        outer_points = [Point(x[0], x[1], x[2]) for x in np.argwhere(outer_s_arr == 1)]
        print(f">> Segment {id_segment} -> processing {len(outer_points)} outer points...") if verbose else None
        if len(outer_points) > outer_points_to_keep:
            random.shuffle(outer_points)
            outer_points = outer_points[::len(outer_points)//outer_points_to_keep]
        print(f">> Segment {id_segment} -> {len(outer_points)} outer points remained.") if verbose else None

        ## Collect centerpoints
        # -> Check if segment is a small square (typically small part of asc or dsc)
        op_min_x, op_max_x = min([p.x for p in outer_points]), max([p.x for p in outer_points])
        op_min_z, op_max_z = min([p.z for p in outer_points]), max([p.z for p in outer_points])
        if (op_max_z - op_min_z)*min_sp < 13 and (op_max_x - op_min_x)*min_sp > 18 and (op_max_x - op_min_x)*min_sp < 55:
            print(f"[WARNING] Small box segment detected. Using only XY erosion.")
            _eroded_xy_seg_s_arr = erode_only_in_xy(seg_s_arr)
            print(np.unique(_eroded_xy_seg_s_arr, return_counts=True))
            centerline_s_arr = skeletonize(_eroded_xy_seg_s_arr)
        else:
            centerline_s_arr = centerline_s_arr
        # -> Get centerline points
        centerline_points = [Point(x[0], x[1], x[2]) for x in np.argwhere(centerline_s_arr == 1)]
        
        ## Reduce centerline points
        print(f">> Segment {id_segment} -> processing {len(centerline_points)} centerline points...") if verbose else None
        centerline_points = centerline_dijkstra(centerline_points, min_sp, distance_to_check=3)
        print(f">> Segment {id_segment} -> {len(centerline_points)} centerline points remained.") if verbose else None
        
        ## Add segment
        print(f">> Segment {id_segment} -> creating segment...") if verbose else None
        segment = Segment(id_segment, centerline_points, outer_points, 
                          keep_only_planes_with_circular_seg=keep_only_planes_with_circular_seg,
                          seg_s_arr_only_for_keep_only_planes_with_circular_seg=seg_s_arr)
        segments.append(segment)
        print(f">> Segment {id_segment} -> {len(segment.centerline_points)} centerline points remained after creation") if verbose else None
        print(f">> Segment {id_segment} -> done.") if verbose else None

    ## Categorize
    print('create_segments: Categorizing...') if verbose else None
    __categorize_segment_planes(segments)
    print('create_segments: Categorizing done.') if verbose else None

    ## Finish
    print('create_segments: Done.') if verbose else None
    return segments

def __categorize_segment_planes(segments: List[Segment]) -> None:
    """Categorize aorta plane
    
    Parameters
    ----------
    segments : List[Segment]
    
    Returns
    -------
    str
        Category of the plane, including: 'DSC', 'ARCH', 'ASC'
    """
    all_centerline_points = []
    for segment in segments:
        all_centerline_points.extend(segment.centerline_points)

    for segment in segments:
        segment.planes_categorization = []
        ## Add based on relative location
        for idx in range(len(segment.planes)):
            point = segment.planes[idx].point
            other_points = [p for p in all_centerline_points if p != point]
            categories_loc = __categorize_aorta_plane_location_based_on_point(point, other_points)
            segment.planes_categorization.append(categories_loc)

    for segment in segments:
        ## Add based on other points classification
        for idx in range(len(segment.planes)):
            point = segment.planes[idx].point
            cats = segment.planes_categorization[idx]
            if len(cats) > 1:
                other_points = [p for p in all_centerline_points if p != point]
                other_categorizations = []
                # Inefficient, but works
                for segment2 in segments:
                    for _idx2 in range(len(segment2.planes)):
                        if idx!=_idx2:
                            other_categorizations.append(segment2.planes_categorization[_idx2])
                segment.planes_categorization[idx] = __categorize_aorta_plane_location_based_on_other_points(point, cats, other_points, other_categorizations)
                # Running twice to be sure. Even more inefficient, but works
                other_categorizations = []
                for segment2 in segments:
                    for _idx2 in range(len(segment2.planes)):
                        if idx!=_idx2:
                            other_categorizations.append(segment2.planes_categorization[_idx2])
                segment.planes_categorization[idx] = __categorize_aorta_plane_location_based_on_other_points(point, cats, other_points, other_categorizations)
        ## Add then based on normals, but a bit more careful
        for idx in range(len(segment.planes)):
            normal = segment.planes[idx].normal
            categories_nor = __categorize_aorta_plane_location_based_on_normal(normal)
            if len(categories_nor) == 1 and categories_nor[0] in segment.planes_categorization[idx]:
                segment.planes_categorization[idx] = categories_nor
        ## Final decision
        for idx in range(len(segment.planes)):
            cats = segment.planes_categorization[idx]
            if len(cats) == 0:
                segment.planes_categorization[idx] = 'ASC'
            elif len(cats) == 1:
                segment.planes_categorization[idx] = cats[0]
            elif len(cats) == 2:
                if 'ASC' in cats:
                    segment.planes_categorization[idx] = 'ASC'
                segment.planes_categorization[idx] = 'DSC'
            else: # 3
                segment.planes_categorization[idx] = 'ASC'

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

    min_y_distance_thres_dsc = -70
    max_z_distance_thres_dsc = 50
    max_y_distance_thres_asc = -min_y_distance_thres_dsc
    min_max_y_distance_thres_arch = 25

    min_y_distance_thres_asc_arch = -30

    ## Certain
    if min_y_distance < min_y_distance_thres_dsc:
        return ['DSC']
    if max_z_distance > max_z_distance_thres_dsc:
        return ['DSC']
    if max_y_distance > max_y_distance_thres_asc:
        return ['ASC']
    if min_y_distance < -min_max_y_distance_thres_arch and max_y_distance > min_max_y_distance_thres_arch:
        return ['ARCH']

    ## Maybes
    if min_y_distance < min_y_distance_thres_asc_arch:
        return ['ASC', 'ARCH']

    ## Uncertain
    return ['ASC', 'DSC', 'ARCH']

def __categorize_aorta_plane_location_based_on_other_points(point, current_categories, all_other_points, all_other_categories):
    point_arr = np.array([point.x, point.y, point.z])
    all_other_points_arr = np.array([[point.x, point.y, point.z] for point in all_other_points])
    xyz_signed_distances_arr = all_other_points_arr - point_arr

    for o_point, o_cats, od in zip(all_other_points, all_other_categories, xyz_signed_distances_arr):
        od_x, od_y, od_z = od

        if len(o_cats) == 1: # if the other is decided
            if 'ARCH' not in o_cats: # if arch is not possible
                if abs(od_x) <= 10 and abs(od_y) <= 10: # roughly on the same-ish line
                    return o_cats
            if 'ARCH' in o_cats and 'ARCH' in current_categories:
                if abs(od_y) <= 3 and abs(od_z) <= 3: # roughly on the same line, pretty strict
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
    THRES_ARCH = 0.80
    THRES_ASC  = 0.20

    n = [round(x,2) for x in normal.as_np_normalized()]

    x, y, z = n[0], n[1], n[2]

    x_pos = abs(x)
    y_pos = abs(y)
    z_pos = abs(z) # z is always positive

    if z_pos > 0.8:
        # ASC or DSC...
        if x > -0.2:
            # ASC or DSC...
            # TODO: NOT SURE ABOUT THIS
            if x > 0.3:
                return ['ASC']
            else:
                if y > 0.2:
                    return ['ASC']
                else:
                    return ['DSC']
        else:
            return ['DSC']
    else:
        return ['ARCH']

