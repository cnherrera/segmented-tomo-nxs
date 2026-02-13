"""
NeXus Tomography for Segmentation Builder

A Python package for creating NeXus-compliant files from segmented tomography data.

Basic Usage
-----------
>>> from nxtomo_segmented_builder import create_nexus_tomo_file
>>> import numpy as np
>>>
>>> data = np.load('segmented_data.npy')
>>> create_nexus_tomo_file(
...     output_file='output.nxs',
...     data=data,
...     voxel_size=[1.0, 1.0, 1.0],
...     title="My Tomography",
...     sample_name="My Sample"
... )

See the documentation for more examples and advanced usage.
"""

from .builder import (
    # Main functions
    create_nexus_tomo_file,
    create_nexus_from_raw_file,
    
    # Helper functions (if users need them)
    load_raw_data,
    create_instrument,
    create_sample,
    create_segmentation_data,
    create_phase_definitions,
    create_characterisation,
    create_process_metadata,
    create_provenance,
)

from .version import __version__

__all__ = [
    # Main API
    'create_nexus_tomo_file',
    'create_nexus_from_raw_file',
    
    # Helpers (optional, for advanced users)
    'load_raw_data',
    'create_instrument',
    'create_sample',
    'create_segmentation_data',
    'create_phase_definitions',
    'create_characterisation',
    'create_process_metadata',
    'create_provenance',
    
    # Metadata
    '__version__',
]

# Package metadata
__author__ = "CH DIAMOND - PEPR DIADEM"
__license__ = "MIT"
