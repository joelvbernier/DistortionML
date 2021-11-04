# DistortionML
Machine Learning approach for quantifying detector distortion fields 

## Running
The script `compute_tth_displacement.py` executes the distortion field calculation based on the single-detector instrument in `resources/`.  It has a progress bar, and plots the distortion field when it completes.

## Parameters
The editable parameters are all located in the following block at the top of the script:
```
# =============================================================================
# %% PARAMETERS
# ============================================================================='
resources_path = './resources'
ref_config = 'reference_instrument.hexrd'

# geometric paramters for source and pinhole (typical TARDIS)
#
# !!! All physical dimensions in mm
#
# !!! This is the minimal set we'd like to do the MCMC over; would like to also
#     include detector translation and at least rotation about its own normal.
rho = 32.                 # source distance
ph_radius = 0.200         # pinhole radius
ph_thickness = 0.100      # pinhole thickness
layer_standoff = 0.150    # offset to sample layer
layer_thickness = 0.01    # layer thickness

# Target voxel size
voxel_size = 0.2
```
The most sensitive parameter is `voxel_size`, which essentially will set the size of the problem, since the number of evaluations will increase quickly for increasing voxel size.  Making `layer_standoff` larger will also increase the total number of voxels contributing for a particular `voxel_size`.
