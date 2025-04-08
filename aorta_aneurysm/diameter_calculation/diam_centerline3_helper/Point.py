"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import random
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import networkx as nx
import itertools


@dataclass
class Point:
    """
    A class representing a point in 3D space.
    """

    x: float
    y: float
    z: float

    _np_cache: np.array = field(default=None)

    value: Union[float, int] = -1 # optional, if you want to store the value of this point

    def as_np(self):
        if self._np_cache is None:
            self._np_cache = np.array([self.x, self.y, self.z])
        return self._np_cache
    
    def as_list(self):
        return [self.x, self.y, self.z]

    def as_nx_node(self):
        return tuple(list(reversed(self.as_list())))
    
    def as_str(self):
        return f"Point({self.x}, {self.y}, {self.z})"

    def distance_to(self, other_point: "Point") -> float:
        """
        Calculate the distance between this point and another point.

        Parameters
        ----------
        other_point: Point
            The other point.

        Returns
        -------
        float
            The distance.
        """
        return np.linalg.norm(self.as_np() - other_point.as_np())

    def is_same_as(self, other_point: "Point") -> bool:
        """
        Check if this point is the same as another point.

        Parameters
        ----------
        other_point: Point
            The other point.

        Returns
        -------
        bool
            True if the points are the same, False otherwise.
        """
        return np.all(self.as_np() == other_point.as_np())


def __get_sensible_starting_point_from_remaining_points(centerline_points, processed):
    rem_points = np.array(centerline_points)[processed == 0]
    return rem_points[np.argmin([p.y for p in rem_points])]


def order_centerline_points(centerline_points: List[Point], distance_threshold : float = 2) -> List[Point]:
    """
    Order the centerline points by distance to each other.

    Parameters
    ----------
    centerline_points: List[Point]
        The list of centerline points.
    distance_threshold: float
        The maximum distance between points to consider them neighbors.
    
    Returns
    -------
    List[Point]
        The ordered list of centerline points.
    """
    ## Get distance between each point and other points
    distances = np.zeros((len(centerline_points), len(centerline_points)))
    for i, p1 in enumerate(centerline_points):
        for j, p2 in enumerate(centerline_points):
            if i != j:
                distances[i, j] = np.linalg.norm( p1.as_np() - p2.as_np() )

    processed = np.zeros(len(centerline_points))

    centerline_points_new = []

    count_no_neighbors = 0
    count_ok = 0

    curr = None
    while sum(processed) < len(centerline_points):
        if curr is None:
            curr = __get_sensible_starting_point_from_remaining_points(centerline_points, processed)
        index = centerline_points.index(curr)
        ## Get distances[index] as a normal 1D array
        others_distances = distances[index]
        others_indexes = np.where(others_distances < distance_threshold)[0]
        others_indexes = [o for o in others_indexes if processed[o] == 0]
        others_indexes = [
            o for o in others_indexes 
            if others_distances[o] > 1e-6 and others_distances[o] < distance_threshold+1e-6
        ]
        others_indexes = sorted(others_indexes, key=lambda x: others_distances[x])
        if len(others_indexes) == 0:
            processed[index] = 1
            count_no_neighbors += 1
            curr = None
            continue

        processed[index] = 1
        count_ok += 1
        centerline_points_new.append(curr)
        curr = centerline_points[others_indexes[0]]
    return centerline_points_new

def remove_close_centerline_points(centerline_points: List[Point], distance_threshold : float = 2) -> List[Point]:
    """
    Remove close centerline points.

    Parameters
    ----------
    centerline_points: List[Point]
        The list of centerline points.
    distance_threshold: float
        The maximum distance between points to consider them neighbors.
    
    Returns
    -------
    List[Point]
        The ordered list of centerline points.
    """
    centerline_points_new = []
    protected = False
    for index in range(len(centerline_points)):
        curr = centerline_points[index]
        prev = centerline_points[index-1] if index > 0 else None
        next = centerline_points[index+1] if index < len(centerline_points)-1 else None
        if protected:
            centerline_points_new.append(curr)
            protected = False
            continue
        if (
            (prev is not None and np.linalg.norm(curr.as_np() - prev.as_np()) > distance_threshold)
            and
            (next is not None and np.linalg.norm(curr.as_np() - next.as_np()) > distance_threshold)
        ):
            centerline_points_new.append(curr)
            protected = False
        else:
            protected = True
            
    return centerline_points_new

def rotate_point(point: Point, R: np.ndarray) -> Point:
    """
    Rotate a point using a rotation matrix.

    Parameters
    ----------
    point: Point
        The point to rotate.
    R: np.ndarray
        The rotation matrix.
    """
    rotated_point_np = R.dot(point.as_np())
    # round
    rotated_point_np = np.round(rotated_point_np, 0).astype(int)
    return Point(rotated_point_np[0], rotated_point_np[1], rotated_point_np[2], value=point.value)


### DIJKSTRA

def points_to_graph(points: List[Point], distance_to_check: int = 1) -> nx.Graph:
    """
    Create graph for dijkstra and more.

    Parameters
    ----------
    points : List[Point]
        The points (e.g. of the centerline).
    distance_to_check: int, optional
        The distance to allow connections between points. Default is 1. Must be between values specified below.

    Returns
    -------
    nx.Graph
        The graph.
    """
    ALLOWED = [1, 2, 3, 4, 5]
    assert distance_to_check in ALLOWED, f"distance_to_check must be in {ALLOWED}"
    iterables_dist = (
        [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] if distance_to_check == 4 else 
        [-4, -3, -2, -1, 0, 1, 2, 3, 4] if distance_to_check == 4 else 
        [-3, -2, -1, 0, 1, 2, 3] if distance_to_check == 3 else 
        [-2, -1, 0, 1, 2]        if distance_to_check == 2 else
        [-1, 0, 1]
    ) 

    G = nx.Graph()

    points_dict = {
        (point.x, point.y ,point.z) : point
        for point in points 
    }

    ## Create graph
    # Nodes
    for point in points:
        x, y, z = point.as_list()
        G.add_node((z, y, x))
    # Edges
    for point in points:
        x, y, z = point.as_list()
        # Consider all possible connections in a 3D grid
        for dx, dy, dz in itertools.product(iterables_dist, repeat=3):
            if dx == dy == dz == 0:
                continue
            other_point = points_dict.get((x + dx, y + dy , z + dz), None)
            if other_point is not None:
                o_x, o_y, o_z = other_point.as_list()                
                G.add_edge((z, y, x), (o_z, o_y, o_x))
                # print(f"Adding nx edge {(z, y, x)} <-> {(o_z, o_y, o_x)} (distances: {dx}, {dy}, {dz})")

    return G

def _dikstra_find_opposite_min_point(
        centerline_points: List[Point], 
        start_point: Point, 
        direction,
        other_direction,
        min_or_max: str,
        # min_or_max_other_direction: str,
        distance_threshold: int = 20,
        ) -> Point:
    """
    Find the opposite point of the centerline.

    Parameters
    ----------
    centerline_points : List[Point]
        The points of the centerline.
    start_point : Point
        The starting point.
    direction : str
        The direction to search: 'x' or 'y' or 'z'.
    other_direction : str
        The other direction to search: 'x' or 'y' or 'z'.
    min_or_max : str
        Are we looking for the other 'min' or 'max'?
    distance_threshold: int, optional
        The distance threshold to consider a point as a candidate. Default is 20.

    Returns
    -------
    Point
        The opposite point.
    """
    if len(centerline_points) <= 1:
        return None

    assert direction in ['x', 'y', 'z'], "direction must be 'x' or 'y' or 'z'"
    assert other_direction in ['x', 'y', 'z'], "other_direction must be 'x' or 'y' or 'z'"
    assert min_or_max in ['min', 'max'], "min_or_max must be 'min' or 'max'"

    min_or_max_func         = min if min_or_max == 'min' else max
    min_or_max_counter_func = max if min_or_max == 'min' else min

    m_point = min_or_max_counter_func(
        centerline_points, 
        key=lambda x: (x.x if direction == 'x' else x.y if direction == 'y' else x.z)
    )

    s2 = start_point.x if other_direction == 'x' else start_point.y if other_direction == 'y' else start_point.z
    m2 = m_point.x if other_direction == 'x' else m_point.y if other_direction == 'y' else m_point.z

    candidates = []
    for point in centerline_points:
        if point.distance_to(start_point) < distance_threshold:
            continue

        d2 = point.x if other_direction == 'x' else point.y if other_direction == 'y' else point.z

        diff1_1 = s2 - m2
        diff1_2 = d2 - m2
        if diff1_1 * diff1_2 < 0: # opposite sides of the max point
            candidates.append(point)

    return min_or_max_func(candidates, key=lambda x: (x.x if direction == 'x' else x.y if direction == 'y' else x.z)) if len(candidates) > 0 else None

def _dijkstra_get_path_minmin_or_maxmax(
        centerline_points: List[Point],
        graphs: List[nx.Graph],
        direction: str,
        other_direction: str,
        min_or_max: str,
):
    argfunc = np.argmin if min_or_max == 'min' else np.argmax

    ds = [p.x for p in centerline_points] if direction == 'x' else [p.y for p in centerline_points] if direction == 'y' else [p.z for p in centerline_points]
    start_point = centerline_points[argfunc(ds)] if len(centerline_points) > 0 else None
    end_point   = _dikstra_find_opposite_min_point(centerline_points, start_point, direction, other_direction, min_or_max, distance_threshold=20)

    ## Collect paths
    paths = []
    for graph in graphs:
        try:
            paths.append( nx.dijkstra_path(graph, source=start_point.as_nx_node(), target=end_point.as_nx_node()) )
        except:
            pass
    ## Return largest
    paths = sorted(paths, key=lambda x: len(x), reverse=True)
    path = paths[0] if len(paths) > 0 else []
    return [Point(*reversed(nx_point)) for nx_point in path]



def _dijkstra_get_path_min_max(
        centerline_points: List[Point],
        graphs: List[nx.Graph],
        direction: str,
        min_to_max: bool = True,
):
    argfunc         = np.argmin if min_to_max else np.argmax
    counter_argfunc = np.argmax if min_to_max else np.argmin


    ds = [p.x for p in centerline_points] if direction == 'x' else [p.y for p in centerline_points] if direction == 'y' else [p.z for p in centerline_points]
    start_point = centerline_points[argfunc(ds)]         if len(centerline_points) > 0 else None
    end_point   = centerline_points[counter_argfunc(ds)] if len(centerline_points) > 0 else None

    ## Collect paths
    paths = []
    for graph in graphs:
        try:
            paths.append( nx.dijkstra_path(graph, source=start_point.as_nx_node(), target=end_point.as_nx_node()) )
        except:
            pass
    ## Return largest
    paths = sorted(paths, key=lambda x: len(x), reverse=True)
    path = paths[0] if len(paths) > 0 else []
    return [Point(*reversed(nx_point)) for nx_point in path]


def centerline_dijkstra(centerline_points_of_segment: List[Point], 
                        voxel_size: float,
                        distance_to_check: int = 1,
                        sanity_check: bool = True,
                        ) -> List[Point]:
    """
    Reduce the points of the centerline to the most important ones, using dijkstra.

    Parameters
    ----------
    centerline_points_of_segment : List[Point]
        The points of the centerline.
    image_shape : tuple
        The shape of the image.
    voxel_size : float
        The voxel size of the image.
    distance_to_check: int, optional
        The distance to allow connections between points. Default is 1.

    Returns
    -------
    List[Point]
        The reduced points of the centerline.
    """
    if len(centerline_points_of_segment) == 0:
        return []

    G  = points_to_graph(centerline_points_of_segment, distance_to_check)
    G3 = points_to_graph(centerline_points_of_segment, distance_to_check+2)

    new_centerline_points = []

    for cc in nx.connected_components(G):
        # print(f"[INFO] Connected component w/ {len(cc)}")
        centerline_points = [Point(*reversed(nx_point)) for nx_point in cc]

        cc_paths = []

        ## -d+
        cc_paths.append( _dijkstra_get_path_min_max(centerline_points, [G, G3], 'z', min_to_max=True) )

        ## -d-
        for d1 in ['x', 'y', 'z']:
            for d2 in ['x', 'y', 'z']:
                if d1 == d2:
                    continue
                if d1 in ['x', 'y']:
                    continue
                cc_paths.append( _dijkstra_get_path_minmin_or_maxmax(centerline_points, [G, G3], d1, d2, 'min') )
                cc_paths.append( _dijkstra_get_path_minmin_or_maxmax(centerline_points, [G, G3], d1, d2, 'max') )

        ## Find the longest cc_paths (covering more centerline points, that is desirable)
        # print(f"\t[INFO] Path lengths: {[len(x) for x in cc_paths]} [orig: {len(centerline_points_of_segment)}]")

        ## Calculate length covered based on coordinates
        cc_length_covered = []
        for path in cc_paths:
            path_legth_coords = 0
            for idx, point in enumerate(path):
                if idx == 0:
                    continue
                prev_point = path[idx-1]
                d = point.distance_to(prev_point)
                path_legth_coords += d
            cc_length_covered.append(path_legth_coords)
        # print(f"\t[INFO] Path ln cvrd: {[x for x in cc_length_covered]} ")

        # Find longest based on cc_length_covered
        path = cc_paths[np.argmax(cc_length_covered)]
        # path = list(sorted(cc_paths, key=lambda x: len(x), reverse=True))[0]

        ## Return the points of the path
        new_centerline_points.extend( path if len(path) > 0 else centerline_points )
    
    ## Sanity check
    if sanity_check:
        if len(centerline_points_of_segment) * (3/4) > len(new_centerline_points): # Changed to (3/4) from (2/3) in 3v4.9
            dist = 0
            t1 = [(p.x, p.y, p.z) for p in centerline_points_of_segment]
            t2 = [(p.x, p.y, p.z) for p in new_centerline_points]
            for p,t in zip(centerline_points,t1):
                if t in t2:
                    continue
                all_dists = [p.distance_to(p2) for p2 in new_centerline_points]
                min_dist = min(all_dists)
                if min_dist > dist:
                    dist = min_dist
            if dist > 70: # Changed to 70 from 80 in 3v4.9
                print(f"[WARNING] >> Dijkstra reduced points too much. Max distance: {dist}. Will return old.")
                return centerline_points_of_segment

    return new_centerline_points


def reduce_similar_centerline_points(centerline_points: List[Point]) -> List[Point]:
    if len(centerline_points) < 10:
        return centerline_points
        
    same_as_before = [False]
    for p_curr, p_prev in zip( centerline_points[1:] , centerline_points[:-1] ):
        # Find long consecutive sections where z is almost constant and x/y are changing maybe slightly
        diff_x, diff_y, diff_z = abs(p_curr.x - p_prev.x), abs(p_curr.y - p_prev.y), abs(p_curr.z - p_prev.z)
        if diff_z <= 3 and diff_x + diff_y <= 2:
            same_as_before.append(True)
        else:
            same_as_before.append(False) 
    
    ## Find sets to merge
    to_merge_tuples = []
    _acc = []
    for idx, s in enumerate(same_as_before):
        if s:
            _acc.append(idx)
        else:
            _acc = []
            continue
        if len(_acc) == 6:
            to_merge_tuples.append(tuple(_acc))
            _acc = []
    
    ## Collect
    to_merge_idxs = [idx for tup in to_merge_tuples for idx in tup]
    handled_idxs = []
    centerline_points_new = []
    for idx, p in enumerate(centerline_points):
        if idx not in to_merge_idxs:
            centerline_points_new.append(p)
        else:
            if idx in handled_idxs:
                continue
            # Find which points to merge
            tup = [t for t in to_merge_tuples if idx in t][0]
            centerline_points_new.append( centerline_points[ tup[3] ] ) 
            # centerline_points_new.append( centerline_points[ tup[4] ] ) 
            handled_idxs.extend(list(tup))
    
    return centerline_points_new