import os

import logging

import h5py

from matplotlib import pyplot as plt

import numpy as np

from tqdm import tqdm

import yaml

from hexrd import constants as ct
from hexrd import instrument
from hexrd.transforms import xfcapi

import pinhole_camera_module as phutil

logger = logging.getLogger(__name__)


# =============================================================================
# %% PARAMETERS
# ============================================================================='
resources_path = './resources'
ref_config = 'reference_intrument.hexrd'

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

# Target voxel size
voxel_size = 0.100


# =============================================================================
# %% OBJECT INSTANTIATION
# =============================================================================

# load instrument and grab the detecor (one for now)
instr = instrument.HEDMInstrument(
    h5py.File(os.path.join(resources_path, ref_config), 'r')
)
det_key, det = next(iter(instr.detectors.items()))  # !!! only one

# generate voxel coordinates in within critial radius
vrho_crit = phutil.compute_voxel_radius(
    layer_standoff, ph_radius, ph_thickness
)


# need the cartesian pixel coordinates
py, px = det.pixel_coords
pixel_xys = np.vstack([px.flatten(), py.flatten()]).T

# also need the reference pixel angles as computed from the origin
ref_ptth, ref_peta = det.pixel_angles()

# generate voxel coordinates and flatten them
vx, vy = np.meshgrid(
    np.linspace(-0.4, 0.4, num=81, endpoint=True),
    np.linspace(-0.4, 0.4, num=81, endpoint=True)
)
vz = np.ones_like(vx)*0.150
vcrds = np.vstack(
    [vx.flatten(),
     vy.flatten(),
     vz.flatten()]
).T


# grand loop over voxels to aggregate pixel angles and contributing voxel count
master_ptth = np.zeros(det.shape, dtype=float)
voxel_count = np.zeros(det.shape, dtype=float)
for iv, vcrd in enumerate(tqdm(vcrds)):
    # need new beam vector from curent voxel coordinate
    new_bv = compute_offset_beam_vector(bhat, rho, np.atleast_2d(vcrd))
    det.bvec = new_bv

    # mask detector pixels
    reduced_rmat = np.ascontiguousarray(det.rmat[:, :2].T)
    mask = pinhole_constraint(pixel_xys, np.array(vcrd),
                              reduced_rmat, det.tvec,
                              ph_radius, ph_thickness)  # no reshape # .reshape(det.shape)

    # compute pixel angles that satisfy the pinhole constraint
    reduced_xys = pixel_xys[mask, :]
    mask = mask.reshape(det.shape)
    ptth = np.nan*np.ones(det.shape)
    angs, _ = xfcapi.detectorXYToGvec(
            reduced_xys, det.rmat, ct.identity_3x3,
            det.tvec, ct.zeros_3, np.array(vcrd),
            beamVec=new_bv)
    ptth[mask] = angs[0]

    master_ptth = np.nansum(
        np.stack([master_ptth, ptth], axis=0),
        axis=0
    )
    voxel_count += mask
mask = np.load('my_ip4_mask.npy')
corr = np.array(master_ptth/voxel_count - ref_ptth, dtype=np.float32)
corr[~mask] = np.nan

fig, ax = plt.subplots()
mappable = ax.imshow(np.degrees(corr), cmap=plt.cm.gnuplot2)
fig.colorbar(mappable)
plt.show()
