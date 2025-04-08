"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import os, copy, json
from pathlib import Path
from typing import List, Union, Tuple, Dict

import SimpleITK as sitk
import pandas as pd

from aorta_aneurysm.util.image import image_flip_with_array, image_resample_to_spacing, image_resample, image_metainformation
from     aorta_aneurysm.util.paths import image_stem
from aorta_aneurysm.util.parallel import run_in_parallel

def _find_needed_flip(img: sitk.Image) -> Dict[str,bool]:
    """
    Find the needed flip for the image [internal function]
    
    Parameters
    ----------
    img : sitk.Image
        The image
        
    Returns
    -------
    dict[str,bool]
        The flip dictionary, with keys 'x', 'y', 'z' and values True/False
    """
    metainfo = image_metainformation(img)
    origin = metainfo["origin"]
    direction = metainfo["direction"]
    rough_direction = metainfo["rough_direction"]
    ## Flip the images if needed to match the model's training data's direction (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    ## For some reason it's apparently y,x,z in the direction matrix but it only affects 1 case, usually x,y are either
    ## flipped together or normal together
    ## The second part of the z-flip doesn't make much sense, because the origin sign shouldn't be important, 
    ## but it fits the data I've seen.
    ## I think it's because FFP position (feet first) and you would expect with position z-direction
    ## to have negative z-origin value, but idk
    flip = { 
        'x': rough_direction[4] < 0, 
        'y': rough_direction[0] < 0, 
        'z': rough_direction[8] < 0 or (rough_direction[8] > 0 and origin[2] > 0)
    }
    return flip

def _flip_if_needed(img: sitk.Image) -> Tuple[sitk.Image, Dict[str,bool]]:
    """
    Flip the image if needed [internal function]
    
    Parameters
    ----------
    img : sitk.Image
        The image
        
    Returns
    -------
    Tuple[sitk.Image, Dict[str,bool]]
        The flipped image and the flip dictionary
    """
    flip = _find_needed_flip(img)
    print(f"FLIP {flip} IMG SIZE {img.GetSize()} IMG SPAC {img.GetSpacing()} IMG ORIG {img.GetOrigin()} IMG DIR {img.GetDirection()}") if os.getenv("DEBUG", "0") != "0" else None
    for axis, should_flip in flip.items():
        if should_flip:
            img = image_flip_with_array(img, axis)
    return img, flip

def _unflip_if_needed(seg: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """
    Unflip the segmentation if needed based on the metadata of the reference [internal function]

    Parameters
    ----------
    seg : sitk.Image
        The image
    ref : sitk.Image
        The reference image

    Returns
    -------
    sitk.Image
        The unflipped segmentation
    """
    flip = _find_needed_flip(ref)
    print(f"UNFLIP {flip} ref SIZE {ref.GetSize()} ref SPAC {ref.GetSpacing()} ref ORIG {ref.GetOrigin()} ref DIR {ref.GetDirection()}") if os.getenv("DEBUG", "0") != "0" else None
    print(f"UNFLIP {flip} seg SIZE {seg.GetSize()} seg SPAC {seg.GetSpacing()} seg ORIG {seg.GetOrigin()} seg DIR {seg.GetDirection()}") if os.getenv("DEBUG", "0") != "0" else None
    for axis, should_flip in reversed(flip.items()): # reversing probably not needed
        if should_flip:
            seg = image_flip_with_array(seg, axis)
    return seg

### ---- SimpleITK ----

def image_to_model_input_image(
        image: Union[sitk.Image,Path],
        save_and_return_None_path: Path = None,
        skip_if_exists: bool = False,
        is_label: bool = False,
        ) -> Union[sitk.Image,None]:
    """Convert a SimpleITK image to a model input image.

    Parameters
    ----------
    image : sitk.Image
        The input image or its path.
    save_and_return_None_path : Path, optional
        If provided, the image will be saved to this path and None will be returned.
    skip_if_exists : bool, optional
        If True, the image will not be processed if the save_and_return_None_path already exists.
    is_label : bool, optional
        Whether the image is a label. Typically this should be False, 
        but it can be set to True if segs are needed on the input format for some reason.

    Returns
    -------
    Union[sitk.Image,None]
        A 3D NIfTI image ready to be fed to the model or None if save_and_return_None_path is provided.
    """
    assert not skip_if_exists or save_and_return_None_path is not None, "skip_if_exists requires save_and_return_None_path"
    ## For saving
    name = save_and_return_None_path.name.split('_')[0] if save_and_return_None_path else None
    json_path = save_and_return_None_path.parent / f"{name}_0000.json" if save_and_return_None_path else None
    if skip_if_exists and save_and_return_None_path.exists() and json_path.exists():
        return None
    if isinstance(image, Path):
        image_copy = sitk.ReadImage(str(image))
    else:
        image_copy = copy.deepcopy(image)
    ## Original direction and origin
    metainfo = image_metainformation(image_copy)
    origin = metainfo["origin"]
    direction = metainfo["direction"]
    size = metainfo["size"]
    spacing = metainfo["spacing"]
    rough_direction = metainfo["rough_direction"]
    origin_sign = metainfo["origin_sign"]
    ## Flip the images if needed to match the model's training data
    image_copy, flip = _flip_if_needed(image_copy)
    ## Remove origin/direction to be sure
    image_copy.SetOrigin((0,0,0))
    image_copy.SetDirection((1,0,0,0,1,0,0,0,1))
    ## Resample
    image_copy = image_resample_to_spacing(image_copy, (0.85,0.85,1.5), is_label=is_label)
    ## Finish
    if save_and_return_None_path:
        sitk.WriteImage(image_copy, str(save_and_return_None_path))
        with open(json_path, "w") as f:
            json.dump({
                "name": name,
                "origin": origin,
                "direction": direction,
                "size": size,
                "spacing": spacing,
                "rough_direction": rough_direction,
                "origin_sign": origin_sign,
                "flip": flip,
            }, f, indent=4)
        return None
    return image_copy

def __no_crash_wrapper___image_to_model_input_image_batch(*args):
    try:
        return image_to_model_input_image(*args)
    except Exception as e:
        print(f"ERROR: {e} - for {args[0]}")
        return None

def image_to_model_input_image_batch(image_paths: List[Path], output_dir: Path, 
                                     num_workers: int = 1, verbose: bool = False, 
                                     is_label: bool = False,
                                     nocrash: bool = False,
                                     file_ending: str = '.nii.gz',
                                     ) -> None:
    """Convert a batch of SimpleITK images to model input images.

    Parameters
    ----------
    image_paths : List[Path]
        The input images.
    output_dir : Path
        The output directory.
    num_workers : int, optional
        The number of workers to use.
    verbose : bool, optional
        Whether to print progress.
    is_label : bool, optional
        Whether the images are labels. Typically this should be False, 
        but it can be set to True if segs are needed on the input format for some reason.
    nocrash : bool, optional
        If set, the program will not crash if an error occurs but will not preprocess problematic case.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ## Run
    if len([x for x in image_paths if '_t1_km' in x.name]):
        _names = [image_path.name.split('_t1_km')[0] for image_path in image_paths if '_t1_km' in image_path.name]
    # TCIA and expected format
    elif len(image_paths[0].name.split('_')) >= 2:
        _names = ['_'.join(image_path.name.split('_')[:-1]) for image_path in image_paths]
    # UNKNOWN
    else:
        _names = [image_stem(image_path) for image_path in image_paths]
    zeros_txt = '_0000' if not is_label else ''
    output_image_paths = [output_dir / f"{_name}{zeros_txt}{file_ending}" for _name in _names]
    work = [(image_path,output_image_path,True,is_label) for image_path,output_image_path in zip(image_paths,output_image_paths)]

    if nocrash:
        if verbose:
            print(f"(running image_to_model_input_image_batch in nocrash mode)")
        run_in_parallel(__no_crash_wrapper___image_to_model_input_image_batch, work, num_workers=num_workers, verbose=verbose)
    else:
        run_in_parallel(image_to_model_input_image, work, num_workers=num_workers, verbose=verbose)

def model_output_seg_to_seg(
        model_seg: Union[sitk.Image,Path], 
        reference: Union[sitk.Image,Path],
        save_and_return_None_path: Path = None,
        resample: bool = True,
        ) -> Union[sitk.Image,None]:
    """Convert a model output segmentation to a segmentation matching the original input data.

    Parameters
    ----------
    model_seg : Union[sitk.Image,Path]
        The model output segmentation image or its path.
    reference : Union[sitk.Image,Path]
        The reference input image or its path.
    save_and_return_None_path : Path, optional
        If provided, the image will be saved to this path and None will be returned.
    resample : bool, optional
        Whether to resample the model output segmentation to match the reference image.
        Normally this should be True, but it can be set to False to keep the spacing of the model output segmentation.

    Returns
    -------
    Union[sitk.Image,None]
        The output image or None if save_and_return_None_path is provided.
    """
    if isinstance(model_seg, Path):
        model_seg = sitk.ReadImage(str(model_seg))
    if isinstance(reference, Path):
        _r = reference
        reference = sitk.ReadImage(str(_r))
        reference_copy = sitk.ReadImage(str(_r))
    else:
        reference_copy = copy.deepcopy(reference)
    ## Nodo the reference_copy
    reference_copy.SetOrigin((0, 0, 0))
    reference_copy.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    ## Resample
    if resample:
        model_seg_final = image_resample(model_seg, reference_copy, is_label=True)
        model_seg_final.SetSpacing(reference.GetSpacing())
    else:
        model_seg_final = copy.deepcopy(model_seg)
    ## Un-nodo the model_seg_final
    model_seg_final.SetDirection(reference.GetDirection())
    model_seg_final.SetOrigin(reference.GetOrigin())
    ## Flip if needed
    model_seg_final = _unflip_if_needed(model_seg_final, reference) # reference, not reference_copy!
    ## Finish
    if save_and_return_None_path:
        sitk.WriteImage(model_seg_final, str(save_and_return_None_path))
        return None
    return model_seg_final

def model_output_seg_to_seg_batch(
        model_seg_paths: List[Path], 
        reference_paths: List[Path], 
        output_dir: Path, 
        resample: bool = True,
        num_workers: int = 1, 
        verbose: bool = False,
        ) -> None:
    """Convert a batch of model output segmentations to segmentations matching the original input data.

    Parameters
    ----------
    model_seg_paths : List[Path]
        The model output segmentations.
    reference_paths : List[Path]
        The reference input images.
    output_dir : Path
        The output directory.
    resample : bool, optional
        Whether to resample the model output segmentations to match the reference images.
        Normally this should be True, but it can be set to False to keep the spacing of the model output segmentations.
    num_workers : int, optional
        The number of workers to use.
    verbose : bool, optional
        Whether to print progress.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ## Run
    work = [(model_seg_path,reference_path,output_dir / model_seg_path.name, resample) for model_seg_path,reference_path in zip(model_seg_paths,reference_paths)]
    
    run_in_parallel(model_output_seg_to_seg, work, num_workers=num_workers, verbose=verbose)
