# NeXus Tomography-Segmentation Builder 

A Python package for creating NeXus-compliant files from segmented tomography data.

## Features

- ✅ Create NeXus NXtomo files from segmented 3D volumes
- ✅ Define multiple phases with physical properties
- ✅ Store thermal and mechanical characterisation data
- ✅ Track processing provenance
- ✅ Calculate phase statistics automatically
- ✅ Full NeXus standard compliance


## How to Use This Package

You have three possible ways to use this tool depending on your setup.

### Option 1 — Install from GitHub (Recommended)

If you can use pip normally:
```
pip install git+https://github.com/cnherrera/segmented-tomo-nxs.git
```

After installation:
```
from nxtomo_segmented_builder import create_nexus_tomo_file
```

If you want a specific version:
```
pip install git+https://github.com/cnherrera/segmented-tomo-nxs.git@v0.1.0
```

### Option 2 — If You Do NOT Have Root Permissions

You can install locally in your user directory:
```
pip install --user git+https://github.com/cnherrera/segmented-tomo-nxs.git
```

If you are using a virtual environment:
```
python -m venv myenv
source myenv/bin/activate
pip install git+https://github.com/cnherrera/segmented-tomo-nxs.git
```

No admin rights required.

### Option 3 — No Git, No Installation (Use builder.py Directly)

If you:
- Do not have git
- Cannot install packages
- Just want to use the script directly

You can:
1. Step 1 — Download the repository as ZIP
   - Go to the GitHub page and click:
   - Code → Download ZIP
   - Unzip it.

2. Step 2 — Copy the file
   - Locate: nxtomo_segmentated_builder/builder.py
   - Copy builder.py into your working directory.

3. Step 3 — Import it directly
   - In your script or notebook:
   ```
   from builder import create_nexus_tomo_file
   ```

That’s it.

⚠️ Important: You still need these Python packages installed:
- numpy
- nexusformat
- h5py

If you cannot install them system-wide:
```
pip install --user numpy nexusformat h5py
```


## Quick Start

### From Raw Binary File

```python
from nxtomo_segmented_builder import create_nexus_from_raw_file

create_nexus_from_raw_file(
    raw_file_path='data.raw',
    output_file='output.nxs',
    shape=(200, 200, 200),
    voxel_size=[0.5, 0.5, 0.5],
    title="Lab Tomography",
    sample_name="Sample A"
)
```

### From NumPy Array

```python
from nxtomo_segmented_builder import create_nexus_tomo_file
import numpy as np

# Your segmented data
data = np.load('segmented_volume.npy')  # Shape: (z, y, x)

# Create NeXus file
create_nexus_tomo_file(
    output_file='output.nxs',
    data=data,
    voxel_size=[1.0, 1.0, 1.0],  # micrometers
    title="My Segmented Tomography",
    sample_name="Foam Sample",
    sample_description="Polyurethane foam"
)
```

### With Phase Definitions

```python
phase_config = {
    0: ("Air/Pore", "Void space"),
    1: ("Polymer Matrix", "Solid phase"),
    2: ("Glass Fiber", "Reinforcement")
}

thermal_props = {
    'conductivity': [0.026, 0.2, 1.5],
    'conductivity_units': 'W/(m*K)'
}

create_nexus_tomo_file(
    output_file='output.nxs',
    data=data,
    voxel_size=[1.0, 1.0, 1.0],
    title="Multi-phase Material",
    sample_name="Composite",
    phase_config=phase_config,
    thermal_props=thermal_props
)
```

## Documentation

Full documentation available at: TBD


## Requirements

Tested on:
- Python ≥ 3.8
- numpy ≥ 1.20.0
- nexusformat ≥ 1.0.0
- h5py ≥ 3.0.0


## License

MIT License - see LICENSE file for details


## Acknowledgments

DIADEM PEPR