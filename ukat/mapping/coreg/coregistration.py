import time, os, copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import dask
import SimpleITK as itk
# import itk
import multiprocessing
#from multiprocessing import shared_memory
#import psutil
import gc

def default_elastix_parameters():
    # See here for default bspline settings and explanation of parameters
    # https://github.com/SuperElastix/ElastixModelZoo/tree/master/models%2Fdefault
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('bspline')
    param_obj.AddParameterMap(parameter_map_bspline) 
    param_obj.SetParameter("FixedImagePyramid", "FixedRecursiveImagePyramid") # "FixedSmoothingImagePyramid"
    param_obj.SetParameter("MovingImagePyramid", "MovingRecursiveImagePyramid") # "MovingSmoothingImagePyramid"
    param_obj.SetParameter("Metric", "AdvancedMeanSquares")
    param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", "50.0")
    param_obj.SetParameter("ErodeMask", "false")
    param_obj.SetParameter("ErodeFixedMask", "false")
    #param_obj.SetParameter("NumberOfResolutions", "4") 
    #param_obj.SetParameter("MaximumNumberOfIterations", "500") # down from 500
    param_obj.SetParameter("MaximumStepLength", "0.1") 
    #param_obj.SetParameter("NumberOfSpatialSamples", "2048")
    #param_obj.SetParameter("BSplineInterpolationOrder", "1")
    #param_obj.SetParameter("FinalBSplineInterpolationOrder", "3")
    #param_obj.SetParameter("DefaultPixelValue", "0")
    param_obj.SetParameter("WriteResultImage", "false")
    
    return param_obj

#Taget and source either 2d (x,y) / 3d (x,y,z)
#Spacing needs to be either single number or list of 2/3 
def coregister(target, source, elastix_model_parameters, spacing, other_sources=None):
    """
    Coregister two arrays and return coregistered + deformation field 
    """
    shape_source = np.shape(source)
    #shape_target = np.shape(target)
    source = itk.GetImageFromArray(np.array(source, np.float32)) 
    source.SetSpacing(spacing)
 
    target = itk.GetImageFromArray(np.array(target, np.float32))
    target.SetSpacing(spacing)

    ## TODO: mask not included TBC
    ## read the source and target images ## removed: NOT USING anymore: SET MOVING IMG; SET FIXED IMAGE
    #elastixImageFilter = itk.ElastixRegistrationMethod.New() 
    
    # Call registration function: NEW
    coregistered, result_transform_parameters = itk.elastix_registration_method(
    source, target,
    parameter_object=elastix_model_parameters)

    # OPTIONAL:print the resulting transform parameters and the elastix bspline paramter file
    # print(result_transform_parameters)

    if other_sources is not None:
        array = itk.GetImageFromArray(np.array(other_sources, np.float32))
        array.SetSpacing(spacing)     
        coreg = itk.transformix_filter(array, result_transform_parameters, log_to_console=False)
        coreg = itk.GetArrayFromImage(coreg)      
    
    ## RUN ELASTIX using ITK-Elastix filters
    coregistered = itk.GetArrayFromImage(coregistered).flatten()

    # Load Transformix Object
    transformix_object = itk.TransformixFilter.New(target)
    transformix_object.SetTransformParameterObject(result_transform_parameters)
    
    # Update object (required)s
    transformix_object.UpdateLargestPossibleRegion()
    # Compute the deformation field
    transformix_object.ComputeDeformationFieldOn() 
    deformation_field = itk.GetArrayFromImage(transformix_object.GetOutputDeformationField()).flatten()

    # Results of Transformation # optional
    # result_image_transformix = transformix_object.GetOutput()
  
    if len(shape_source) == 2: # 2D
        deformation_field = np.reshape(deformation_field,(shape_source[0], shape_source[1], 2)) 
    else: #3D 
        deformation_field = np.reshape(deformation_field,(shape_source[0], shape_source[1], shape_source[2], 3)) 

    if other_sources is not None:
        return coregistered, deformation_field, coreg
    return coregistered, deformation_field

def mdr(array, pixel_spacing, signal_model, *args, other=None, max_it=5, largest_deformation=1):
    # (x,y,t)
    coreg = array.copy()
    defo = np.zeros(array.shape + (2,))

    params = default_elastix_parameters()
    if other is not None:
        other_coreg = np.zeros(array.shape)

    for it in range(max_it):
        target = signal_model(coreg, *args).get_fit_signal()
        for k in range(array.shape[-1]):
            if other is None:
                coreg[...,k], defo[...,k,:] = coregister(target[...,k], array[...,k], params, pixel_spacing)
            else:
                coreg[...,k], defo[...,k,:], other_coreg[...,k] = coregister(target[...,k], array[...,k], params, pixel_spacing, other=other)

        defo_max = np.amax(np.linalg.norm(defo, axis=-1))
        if defo_max<largest_deformation:
            return coreg, defo
        
    return coreg, defo

#Save gifs put some progress bars