#!/usr/bin/env python
# coding: utf-8


"""
NeXus NXtomo File Builder for Segmented Tomography Data

This module provides functions to create NeXus-compliant files for segmented
tomography data with customizable metadata and properties.

Author: DIAMOND - PEPR DIADEM
Date: 2026-02-06
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from nexusformat.nexus import *


# =====================================================================
# DATA LOADING
# =====================================================================

def load_raw_data(file_path: str, shape: Tuple[int, int, int], dtype: Union[str, np.dtype] = np.uint8) -> np.ndarray:
    """
    Load raw binary tomography data.
    
    Parameters
    ----------
    file_path : str
        Path to the raw binary file
    shape : tuple of int
        Shape of the data (z, y, x)
    dtype : str or numpy.dtype
        Data type of the raw file
        
    Returns
    -------
    np.ndarray
        Loaded data with specified shape
    """
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=dtype)
        raw_data = raw_data.reshape(shape)
    return raw_data


# =====================================================================
# METADATA BUILDERS
# =====================================================================

def create_instrument(source_type: str = "X-ray tube", source_name: str = "Laboratory micro-CT", detector_description: str = "Lab detector",
    voxel_size: List[float] = [1.0, 1.0, 1.0], exposure_time: Optional[float] = None, n_projections: Optional[int] = None) -> NXinstrument:
    """
    Create NXinstrument group with source and detector information.
    
    Parameters
    ----------
    source_type : str
        Type of radiation source (e.g., "X-ray tube", "Synchrotron")
    source_name : str
        Name/model of the source
    detector_description : str
        Description of the detector
    voxel_size : list of float
        Voxel size [z, y, x] in micrometers
    exposure_time : float, optional
        Exposure time in seconds
    n_projections : int, optional
        Number of projections acquired
        
    Returns
    -------
    NXinstrument
        Populated instrument group
    """
    instrument = NXinstrument()
    
    # Source
    source = NXsource()
    source.type = source_type
    source.name = source_name
    source.probe = "x-ray" if "ray" in source_type.lower() else "neutron"
    instrument['source'] = source
    
    # Detector
    detector = NXdetector()
    detector.description = detector_description
    detector.x_pixel_size = NXfield(voxel_size[2], units="micrometer")
    detector.y_pixel_size = NXfield(voxel_size[1], units="micrometer")
    if exposure_time is not None:
        detector.exposure_time = NXfield(exposure_time, units="s")
    if n_projections is not None:
        detector.number_of_projections = n_projections
    instrument['detector'] = detector
    
    return instrument


def create_sample(name: str, description: str, shape: Tuple[int, int, int], voxel_size: List[float], additional_metadata: Optional[Dict] = None) -> NXsample:
    """
    Create NXsample group with sample information.
    
    Parameters
    ----------
    name : str
        Sample name
    description : str
        Sample description
    shape : tuple of int
        Data shape (z, y, x) in pixels
    voxel_size : list of float
        Voxel size [z, y, x] in micrometers
    additional_metadata : dict, optional
        Additional metadata to add to sample
        
    Returns
    -------
    NXsample
        Populated sample group
    """
    sample = NXsample()
    sample.name = name
    sample.description = description
    sample.voxel_size = NXfield(voxel_size, units="micrometer")
    sample.dimension_pixels = shape
    
    # Calculate physical dimensions
    physical_dims = [shape[i] * voxel_size[i] for i in range(3)]
    sample.dimension_physical = NXfield(physical_dims, units="micrometer")
    
    # Add any additional metadata
    if additional_metadata:
        for key, value in additional_metadata.items():
            if isinstance(value, dict) and 'value' in value and 'units' in value:
                sample[key] = NXfield(value['value'], units=value['units'])
            else:
                sample[key] = value
    
    return sample


def create_segmentation_data(data: np.ndarray, voxel_size: List[float], data_name: str = "segmentation") -> NXdata:
    """
    Create NXdata group for segmentation with proper axes.
    
    Parameters
    ----------
    data : np.ndarray
        Segmented data array
    voxel_size : list of float
        Voxel size [z, y, x] in micrometers
    data_name : str
        Name for the data group
        
    Returns
    -------
    NXdata
        Segmentation data group with axes
    """
    shape = data.shape
    seg = NXdata(name=data_name)
    
    # Main data
    seg.data = NXfield(data.astype(np.uint8), units="dimensionless")
    seg.data.long_name = "Segmentation labels"
    
    # Create spatial axes
    z_axis = NXfield(np.arange(shape[0]) * voxel_size[0],
        units="micrometer",
        long_name="Z position")
    y_axis = NXfield(np.arange(shape[1]) * voxel_size[1],
        units="micrometer",
        long_name="Y position")
    x_axis = NXfield(np.arange(shape[2]) * voxel_size[2],
        units="micrometer",
        long_name="X position")
    
    seg['z'] = z_axis
    seg['y'] = y_axis
    seg['x'] = x_axis
    seg.attrs['axes'] = ['z', 'y', 'x']
    seg.attrs['signal'] = 'data'
    
    return seg


def create_phase_definitions(data: np.ndarray, voxel_size: List[float],
    phase_config: Optional[Dict[int, Tuple[str, str]]] = None) -> NXcollection:
    """
    Create phase definitions with statistics.
    
    Parameters
    ----------
    data : np.ndarray
        Segmented data
    voxel_size : list of float
        Voxel size [z, y, x] in micrometers
    phase_config : dict, optional
        Configuration dict: {label: (name, description)}
        If None, generic names will be used
        
    Returns
    -------
    NXcollection
        Phases collection with statistics
    """
    phases = NXcollection(name="phases")
    labels_present = np.unique(data)
    voxel_volume_um3 = np.prod(voxel_size)
    
    for label in labels_present:
        p = NXcollection()
        p.index = int(label)
        
        # Use config or generate generic name
        if phase_config and label in phase_config:
            p.name = phase_config[label][0]
            p.description = phase_config[label][1]
        else:
            p.name = f"Phase {label}"
            p.description = "User-defined phase"
        
        # Calculate statistics
        voxel_count = int(np.sum(data == label))
        p.volume_fraction = float(voxel_count / data.size)
        p.voxel_count = voxel_count
        p.physical_volume = NXfield(
            voxel_count * voxel_volume_um3,
            units="micrometer^3"
        )
        
        phases[f"phase_{int(label)}"] = p
    
    return phases


def create_characterisation(labels: np.ndarray,
    thermal_props: Optional[Dict[str, List]] = None,
    mechanical_props: Optional[Dict[str, List]] = None,
    custom_props: Optional[Dict[str, Dict]] = None) -> NXcollection:
    """
    Create characterisation data for phases.
    
    Parameters
    ----------
    labels : np.ndarray
        Array of unique labels in the segmentation
    thermal_props : dict, optional
        Thermal properties: {'conductivity': [values], ...}
        Units should be specified as {'property': [values], 'units': 'unit_string'}
    mechanical_props : dict, optional
        Mechanical properties: {'young_modulus': [values], ...}
    custom_props : dict, optional
        Custom properties: {'category_name': {'property': values, 'units': units}}
        
    Returns
    -------
    NXcollection
        Characterisation collection
    """
    char = NXcollection(name="characterisation")
    
    # Thermal properties
    if thermal_props:
        thermal = NXcollection()
        thermal.description = "Thermal properties per phase"
        for prop_name, values in thermal_props.items():
            if prop_name != 'units':
                units = thermal_props.get(f'{prop_name}_units', 'dimensionless')
                thermal[prop_name] = NXfield(values, units=units)
        thermal.phase_labels = labels
        char['thermal'] = thermal
    
    # Mechanical properties
    if mechanical_props:
        mech = NXcollection()
        mech.description = "Mechanical properties per phase"
        for prop_name, values in mechanical_props.items():
            if prop_name != 'units':
                units = mechanical_props.get(f'{prop_name}_units', 'dimensionless')
                mech[prop_name] = NXfield(values, units=units)
        mech.phase_labels = labels
        char['mechanical'] = mech
    
    # Custom properties
    if custom_props:
        for category_name, properties in custom_props.items():
            custom_coll = NXcollection()
            custom_coll.description = properties.get('description', f'{category_name} properties')
            for prop_name, value in properties.items():
                if prop_name not in ['description', 'units']:
                    units = properties.get('units', {}).get(prop_name, 'dimensionless')
                    custom_coll[prop_name] = NXfield(value, units=units)
            custom_coll.phase_labels = labels
            char[category_name] = custom_coll
    
    return char


def create_process_metadata(program: str, algorithm: str, version: Optional[str] = None,
    parameters: Optional[Dict] = None, date: Optional[str] = None) -> NXprocess:
    """
    Create segmentation process metadata.
    
    Parameters
    ----------
    program : str
        Name of segmentation software
    algorithm : str
        Algorithm/method used
    version : str, optional
        Software version
    parameters : dict, optional
        Processing parameters
    date : str, optional
        Processing date (ISO format). If None, uses current time
        
    Returns
    -------
    NXprocess
        Process metadata group
    """
    process = NXprocess()
    process.program = program
    process.algorithm = algorithm
    
    if version:
        process.version = version
    
    process.date = date if date else datetime.now().isoformat()
    
    if parameters:
        param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
        process.parameters = param_str
    
    return process


def create_provenance(raw_file_path: str, raw_dtype: Union[str, np.dtype], additional_info: Optional[Dict] = None) -> NXcollection:
    """
    Create provenance metadata.
    
    Parameters
    ----------
    raw_file_path : str
        Original data file path
    raw_dtype : str or dtype
        Data type of raw file
    additional_info : dict, optional
        Additional provenance information
        
    Returns
    -------
    NXcollection
        Provenance metadata
    """
    prov = NXcollection(name="provenance")
    prov.original_raw_path = str(raw_file_path)
    prov.raw_dtype = str(raw_dtype)
    prov.file_type = "binary raw"
    prov.creation_date = datetime.now().isoformat()
    prov.nexusformat_version = nxversion
    prov.numpy_version = np.__version__
    
    if additional_info:
        for key, value in additional_info.items():
            prov[key] = value
    
    return prov



# =====================================================================
# MAIN BUILDER FUNCTION
# =====================================================================

def create_nexus_tomo_file(
    output_file: str,
    data: np.ndarray,
    voxel_size: List[float],
    # Entry metadata
    title: str,
    definition: str = "NXtomo",
    # Sample info
    sample_name: str = "Sample",
    sample_description: str = "Segmented tomography sample",
    sample_metadata: Optional[Dict] = None,
    # Instrument info
    source_type: str = "X-ray tube",
    source_name: str = "Laboratory micro-CT",
    detector_description: str = "Lab detector",
    exposure_time: Optional[float] = None,
    n_projections: Optional[int] = None,
    # Phase definitions
    phase_config: Optional[Dict[int, Tuple[str, str]]] = None,
    # Characterisation
    thermal_props: Optional[Dict] = None,
    mechanical_props: Optional[Dict] = None,
    custom_char_props: Optional[Dict] = None,
    # Process info
    segmentation_program: Optional[str] = None,
    segmentation_algorithm: Optional[str] = None,
    segmentation_version: Optional[str] = None,
    segmentation_parameters: Optional[Dict] = None,
    # Provenance
    raw_file_path: Optional[str] = None,
    raw_dtype: Union[str, np.dtype] = np.uint8,
    provenance_info: Optional[Dict] = None,
    # Options
    include_phases: bool = True,
    include_provenance: bool = True,
    overwrite: bool = False,
    print_tree: bool = False) -> str:
    """
    Create a complete NeXus NXtomo file for segmented tomography data.
    
    Parameters
    ----------
    output_file : str
        Output NeXus file path (.nxs)
    data : np.ndarray
        Segmented data array (z, y, x)
    voxel_size : list of float
        Voxel size [z, y, x] in micrometers
    title : str
        Entry title
    definition : str
        NeXus definition (default: "NXtomo")
    sample_name : str
        Name of the sample
    sample_description : str
        Sample description
    sample_metadata : dict, optional
        Additional sample metadata
    source_type : str
        Radiation source type
    source_name : str
        Source name/model
    detector_description : str
        Detector description
    exposure_time : float
        Exposure time in seconds
    n_projections : int, optional
        Number of projections
    phase_config : dict, optional
        Phase configuration: {label: (name, description)}
    thermal_props : dict, optional
        Thermal properties per phase
    mechanical_props : dict, optional
        Mechanical properties per phase
    custom_char_props : dict, optional
        Custom characterisation properties
    segmentation_program : str, optional
        Segmentation software name
    segmentation_algorithm : str, optional
        Segmentation algorithm/method
    segmentation_version : str, optional
        Software version
    segmentation_parameters : dict, optional
        Processing parameters
    raw_file_path : str, optional
        Original raw data file path
    raw_dtype : str or dtype
        Raw data type
    provenance_info : dict, optional
        Additional provenance information
    include_phases : bool
        Include phase definitions (default: True)
    include_provenance : bool
        Include provenance metadata (default: True)
    overwrite : bool
        Overwrite existing file (default: False)
    print_tree : bool
        Print the structure (tree) of the output file 
        
    Returns
    -------
    str
        Path to created file
        
    Raises
    ------
    FileExistsError
        If output file exists and overwrite=False
    ValueError
        If data shape doesn't match voxel_size dimensions
    """
    # Validate inputs
    if len(voxel_size) != 3:
        raise ValueError("voxel_size must have 3 elements [z, y, x]")
    
    if data.ndim != 3:
        raise ValueError("data must be 3D array")
    
    output_path = Path(output_file)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"File {output_file} already exists. Use overwrite=True to replace."
        )
    
    # Create root structure
    root = NXroot()
    entry = NXentry()
    root['entry'] = entry
    
    # Entry metadata
    entry.title = title
    entry.start_time = datetime.now().isoformat()
    entry.end_time = datetime.now().isoformat()
    entry.definition = definition
    entry.version = "1.0"
    
    # Instrument
    instrument = create_instrument(
        source_type=source_type,
        source_name=source_name,
        detector_description=detector_description,
        voxel_size=voxel_size,
        exposure_time=exposure_time,
        n_projections=n_projections
    )
    entry['instrument'] = instrument
    
    # Sample
    sample = create_sample(
        name=sample_name,
        description=sample_description,
        shape=data.shape,
        voxel_size=voxel_size,
        additional_metadata=sample_metadata
    )
    entry['sample'] = sample
    
    # Segmentation data (main data group)
    seg_data = create_segmentation_data(
        data=data,
        voxel_size=voxel_size
    )
    entry['data'] = seg_data
    
    # Phase definitions
    if include_phases:
        phases = create_phase_definitions(
            data=data,
            voxel_size=voxel_size,
            phase_config=phase_config
        )
        entry['phases'] = phases
    
    # Characterisation
    if (thermal_props or mechanical_props or custom_char_props):
        labels = np.unique(data)
        char = create_characterisation(
            labels=labels,
            thermal_props=thermal_props,
            mechanical_props=mechanical_props,
            custom_props=custom_char_props
        )
        entry['characterisation'] = char
    
    # Process metadata
    if segmentation_program:
        process = create_process_metadata(
            program=segmentation_program,
            algorithm=segmentation_algorithm or "Not specified",
            version=segmentation_version,
            parameters=segmentation_parameters
        )
        entry['segmentation_process'] = process
    
    # Provenance
    if include_provenance: # and raw_file_path is not None:
        prov = create_provenance(
            raw_file_path=raw_file_path,
            raw_dtype=raw_dtype,
            additional_info=provenance_info
        )
        entry['provenance'] = prov
    
    # Save file
    mode = 'w' if overwrite else 'w-'
    root.save(str(output_file), mode=mode)

    # Verify the file structure
    if print_tree:
        root = nxload(output_file)
        print(root.tree)

    
    return str(output_path.absolute())


# =====================================================================
# CONVENIENCE FUNCTION WITH FILE LOADING
# =====================================================================

def create_nexus_from_raw_file(
    raw_file_path: str,
    output_file: str,
    shape: Tuple[int, int, int],
    voxel_size: List[float],
    raw_dtype: Union[str, np.dtype] = np.uint8,
    **kwargs) -> str:
    """
    Load raw data and create NeXus file in one step.
    
    Parameters
    ----------
    raw_file_path : str
        Path to raw binary file
    output_file : str
        Output NeXus file path
    shape : tuple of int
        Data shape (z, y, x)
    voxel_size : list of float
        Voxel size [z, y, x] in micrometers
    raw_dtype : str or dtype
        Raw data type
    **kwargs
        Additional arguments passed to create_nexus_tomo_file()
        
    Returns
    -------
    str
        Path to created file
    """
    # Load data
    data = load_raw_data(raw_file_path, shape, raw_dtype)
    
    # Add raw file path to kwargs if not already there
    if 'raw_file_path' not in kwargs:
        kwargs['raw_file_path'] = raw_file_path
    if 'raw_dtype' not in kwargs:
        kwargs['raw_dtype'] = raw_dtype
    
    # Create file
    return create_nexus_tomo_file(
        output_file=output_file,
        data=data,
        voxel_size=voxel_size,
        **kwargs
    )
