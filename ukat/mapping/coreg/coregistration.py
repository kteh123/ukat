import time, os, copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import dask
import SimpleITK as itk
import multiprocessing
#from multiprocessing import shared_memory
#import psutil
import gc

def _coregister(target, source, elastix_model_parameters, spacing, log, mask):
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
    
    ## RUN ELASTIX using ITK-Elastix filters
    coregistered = itk.GetArrayFromImage(coregistered).flatten()

    # Load Transformix Object
    transformix_object = itk.TransformixFilter.New(target)
    transformix_object.SetTransformParameterObject(result_transform_parameters)
    
    # Update object (required)
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

    return coregistered, deformation_field

def _elastix2dict(elastix_model_parameters):
    """
    Hack to allow parallel processing
    """
    list_dictionaries_parameters = []
    for index in range(elastix_model_parameters.GetNumberOfParameterMaps()):
        parameter_map = elastix_model_parameters.GetParameterMap(index)
        one_parameter_map_dict = {}
        for i in parameter_map:
            one_parameter_map_dict[i] = parameter_map[i]
        list_dictionaries_parameters.append(one_parameter_map_dict)
    return list_dictionaries_parameters

def _dict2elastix(list_dictionaries_parameters):
    """
    Hack to allow parallel processing
    """
    elastix_model_parameters = itk.ParameterObject.New()
    for one_map in list_dictionaries_parameters:
        elastix_model_parameters.AddParameterMap(one_map)
    return elastix_model_parameters

def get_elastix_parameters(self, output_directory=os.getcwd(),
                               base_file_name='Elastix_Parameters',
                               export=False):
        """
        Returns a itk.ParameterObject with the elastix registration parameters.
        Parameters
        ----------
        output_directory : string, optional
            Path to the folder that will contain the TXT file to be saved,
            if export=True.
        base_file_name : string, optional
            Filename of the exported TXT. This code appends the extension.
            Eg., base_file_name = 'Elastix_Parameters' will result in
            'Elastix_Parameters.txt'.
        export : bool, optional
            If True (default is False), the elastix registration parameters
            are saved as "Elastix_Parameters.txt" in the 'output_directory'.
        Returns
        -------
        self._elastix_params : itk.ParameterObject
            Private attribute with the elastix registration parameters
            that is returned if this getter function is called.
        """
        if export:
            file_path = os.path.join(output_directory,
                                     base_file_name + ".txt")
            text_file = open(file_path, "w")
            print(self._elastix_params, file=text_file)
            text_file.close()
        return self._elastix_params