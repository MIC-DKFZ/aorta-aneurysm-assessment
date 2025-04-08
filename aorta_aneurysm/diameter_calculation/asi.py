"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0

ASI (Aortic Size Index) calculation
from https://www.sciencedirect.com/science/article/pii/S0003497505010313
"""

def asi_calculate(diameter_asc_cm: float, height_m: float, weight_kg: float) -> float:
    """
    Calculate asi (Aortic Size Index) from aortic diameter and weight+height
    Only works for ascending part!
    :param diameter_asc_cm: aortic diameter in cm
    :param height_m: aortic height in m
    :param weight_kg: weight in kg
    :return: asi
    """

    if (
        diameter_asc_cm < 1e-12 or height_m < 1e-12 or weight_kg < 1e-12 or
        not isinstance(diameter_asc_cm, (int, float)) or 
        not isinstance(height_m, (int, float)) or
        not isinstance(weight_kg, (int, float))
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
         (height_m > 0 and height_m <= 2.6) and
         (weight_kg > 0 and weight_kg <= 300)
    )):
        return -1

    ## Dubois formula
    height_m = float(height_m)
    weight_kg = float(weight_kg)
    bsa = 0.20247 * pow(weight_kg, 0.425) * pow(height_m, 0.725)

    return diameter_asc_cm / bsa

def asi_classify(asi: float) -> str:
    """
    Classify asi into categories
    :param asi: asi value
    :return: asi category
    """
    asi = round(asi,2)

    # THRESHOLDS FROM THIS PART:
    # In particular, we found that using ASI, 
    # patients could be stratified into three categories of risk (Fig 3). 
    # Those with ASI less than 2.75 cm/m2 are at low risk for negative events, 
    # with a yearly incidence of approximately 4%, 
    # those with ASI between 2.75 and 4.25 cm/m2 are at moderate risk with 
    # yearly incidence of approximately 8%, whereas those with ASI above 4.25 cm/m2 have yearly 
    # rates of rupture, dissection, or death as high as 20% to 25%. 

    if asi < 0.01:
        return "Error calculating"
    elif asi < 2.75:
        return "4% average yearly risk of complications"
    elif asi <= 4.25:
        return "8% average yearly risk of complications"
    else:
        return "20% average yearly risk of complications"

