"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk

from aorta_aneurysm.util.array import keep_at_most_2_axial_cc, keep_largest_cc

def get_xyz_array_from_image(image : sitk.Image) -> np.ndarray:
    """
    Get the XYZ array from the image

    Parameters
    ----------
    image : sitk.Image
        The image

    Returns
    -------
    np.ndarray
        The XYZ array
    """
    arr = sitk.GetArrayFromImage(image)
    arr = np.transpose(arr, (2, 1, 0))
    return arr

def get_image_from_xyz_array(arr : np.ndarray) -> sitk.Image:
    """
    Get the image from the XYZ array

    Parameters
    ----------
    arr : np.ndarray
        The XYZ array

    Returns
    -------
    sitk.Image
        The image
    """
    return sitk.GetImageFromArray( np.transpose(arr, (2, 1, 0)) )

def image_metainformation(image: Union[sitk.Image,Path]) -> dict:
    """
    Get the metainformation of the image, quickly.

    Parameters
    ----------
    image_path : Union[sitk.Image,Path]
        The image or its path

    Returns
    -------
    dict
        The metainformation: size, spacing, origin, direction, rough_direction, origin_sign
    """
    ## Read metainformation
    if isinstance(image, sitk.Image):
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
    else:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(image))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        size = reader.GetSize()
        spacing = reader.GetSpacing()
        origin = reader.GetOrigin()
        direction = reader.GetDirection()
    ## Rough direction
    rough_direction = [round(x,1) for x in direction]
    for i,v in enumerate(rough_direction):
        if v >= -0.2 and v <= 0.2:
            rough_direction[i] = 0
        elif v >= 0.8 and v <= 1.2:
            rough_direction[i] = 1
        elif v <= -0.8 and v >= -1.2:
            rough_direction[i] = -1
        else:
            print(f"WEIRD DIRECTION VALUE: {v}")
    rough_direction = tuple(rough_direction)
    ## Origin sign
    origin_sign = tuple([1 if x > 0 else -1 for x in origin])
    ## Rough spacing
    rough_spacing = [round(x,2) for x in spacing]
    ## Finish
    return {
        "size": size,
        "spacing": spacing,
        "rough_spacing": rough_spacing,
        "origin": origin,
        "origin_sign": origin_sign,
        "direction": direction,
        "rough_direction": rough_direction,
    }

def image_resample(img: sitk.Image, ref: sitk.Image, is_label: bool) -> sitk.Image:
    """
    Resample the image to the reference image

    Parameters
    ----------
    img : sitk.Image
        The image
    ref : sitk.Image
        The reference image
    is_label : bool
        Whether the image is a label image

    Returns
    -------
    sitk.Image
        The resampled image
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(img)

def image_resample_to_spacing(
        img: Union[sitk.Image,Path,str], 
        new_spacing: tuple, 
        is_label: bool,
        save_and_return_None_path: Union[Path,str] = None
    ) -> Union[sitk.Image,None]:
    """
    Resample the image to the new spacing
    
    Parameters
    ----------
    img : Union[sitk.Image,Path,str]
        The image or its path
    new_spacing : tuple
        The new spacing
    is_label : bool
        Whether the image is a label image
    save_and_return_None_path : Union[Path,str], optional
        The path to save the resampled image. If not None then the image is saved and None is returned.

    Returns
    -------
    Union[sitk.Image,None]
        The resampled image or None if save_and_return_None_path is not None
    """
    if isinstance(img, Path) or isinstance(img, str):
        img = sitk.ReadImage(str(img))
    # Get the original spacing
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0) # was img.GetPixelIDValue()
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    if save_and_return_None_path is not None:
        sitk.WriteImage(resampler.Execute(img), str(save_and_return_None_path))
        return None
    else:
        return resampler.Execute(img)

def image_flip_with_array(image: sitk.Image, axis: str) -> sitk.Image:
    """
    Flip the image along the specified axis

    Parameters
    ----------
    image : sitk.Image
        The input image
    axis : str
        The axis to flip along ('x', 'y', or 'z')

    Returns
    -------
    sitk.Image
        The flipped image
    """
    arr = sitk.GetArrayFromImage(image)
    if axis == 'x':
        arr = arr[:, ::-1, :]
    elif axis == 'y':
        arr = arr[:, :, ::-1]
    elif axis == 'z':
        arr = arr[::-1, :, :]
    else:
        raise ValueError(f"Unexpected axis: {axis}")
    flipped_image = sitk.GetImageFromArray(arr)
    flipped_image.SetSpacing(image.GetSpacing())
    flipped_image.SetOrigin(image.GetOrigin())
    flipped_image.SetDirection(image.GetDirection())
    return flipped_image

def image_keep_at_most_2_axial_cc(seg: sitk.Image) -> sitk.Image:
    """
    Keep at most 2 axial connected components in the image

    Parameters
    ----------
    seg : sitk.Image
        The input image

    Returns
    -------
    sitk.Image
        The image with at most 2 axial connected components
    """
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_arr = keep_at_most_2_axial_cc(seg_arr, destructive=True)
    new_seg = sitk.GetImageFromArray(seg_arr)
    new_seg.SetSpacing(seg.GetSpacing())
    new_seg.SetOrigin(seg.GetOrigin())
    new_seg.SetDirection(seg.GetDirection())
    return new_seg

def image_keep_largest_cc(seg: sitk.Image, connectivity=6) -> sitk.Image:
    """
    Keep the largest connected component in the image

    Parameters
    ----------
    seg : sitk.Image
        The input image

    Returns
    -------
    sitk.Image
        The image with only the largest connected component
    """
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_arr = keep_largest_cc(seg_arr, destructive=True, connectivity=connectivity)
    new_seg = sitk.GetImageFromArray(seg_arr)
    new_seg.SetSpacing(seg.GetSpacing())
    new_seg.SetOrigin(seg.GetOrigin())
    new_seg.SetDirection(seg.GetDirection())
    return new_seg

def image_seg_binarize(seg: sitk.Image, label: int) -> sitk.Image:
    """
    Keep only the specified label in the image

    Parameters
    ----------
    seg : sitk.Image
        The input image
    label : int
        The label to keep

    Returns
    -------
    sitk.Image
        The image with only the specified label
    """
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_arr = (seg_arr == label).astype(np.uint8)
    new_seg = sitk.GetImageFromArray(seg_arr)
    new_seg.SetSpacing(seg.GetSpacing())
    new_seg.SetOrigin(seg.GetOrigin())
    new_seg.SetDirection(seg.GetDirection())
    return new_seg