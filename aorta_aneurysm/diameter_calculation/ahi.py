"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0

AHI (Aortic Height Index) calculation
from https://www.jtcvs.org/article/S0022-5223(17)32769-1/pdf
"""

def ahi_calculate(diameter_asc_cm: float, height_m: float) -> float:
    """
    Calculate AHI (Aortic Height Index) from aortic diameter and height
    Only works for ascending part!
    :param diameter_asc_cm: aortic diameter in cm
    :param height_m: aortic height in m
    :return: AHI
    """

    if (
        diameter_asc_cm < 1e-12 or height_m < 1e-12 or 
        not isinstance(diameter_asc_cm, (int, float)) or not isinstance(height_m, (int, float))
    ):
        return -1
    
    ## Convert regular mistakes
    if diameter_asc_cm > 12.0:
        diameter_asc_cm = diameter_asc_cm / 10.0 # mm to cm
    if height_m > 2.6:
        height_m = height_m / 100.0

    ## Sanity check
    if (not(
         (diameter_asc_cm > 0 and diameter_asc_cm <= 12.0) and
         (height_m > 0 and height_m <= 2.6)
    )):
        return -1

    return diameter_asc_cm / height_m

def ahi_classify(ahi: float) -> str:
    """
    Classify AHI into categories
    :param ahi: AHI value
    :return: AHI category
    """
    ahi = round(ahi,2)

    if ahi < 0.01:
        return "Error calculating"
    elif ahi <= 2.43:
        return "4% average yearly risk of complications"
    elif ahi < 3.21:
        return "7% average yearly risk of complications"
    elif ahi < 4.10:
        return "12% average yearly risk of complications"
    else:
        return "18% average yearly risk of complications"

