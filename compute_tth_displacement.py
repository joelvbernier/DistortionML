import os

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import logging
import time
import h5py

from matplotlib import pyplot as plt

import numpy as np

import numba

from skimage import io

from tqdm import tqdm

from hexrd import constants as ct
from hexrd import gridutil
from hexrd import instrument
from hexrd import material
from hexrd.transforms import xfcapi
from hexrd.utils.concurrent import distribute_tasks
from hexrd import valunits

import pinhole_camera_module as phutil

logger = logging.getLogger(__name__)


# =============================================================================
# %% PARAMETERS
# ============================================================================='
resources_path = './resources'
ref_config = 'reference_instrument.hexrd'

det_key = 'IMAGE-PLATE-4'

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
layer_thickness = 0.025   # layer thickness

# Target voxel size
voxel_size = 0.025

matl_fname = 'simulation_materials.h5'

dmin = 0.5

# for powder
fwhm_powder = 1.

# for Laue
energy_cutoffs = [5, 20]  # keV, for Laue
grain_params = np.hstack(
    [[0., 0., np.radians(11)],
     ct.zeros_3,
     ct.identity_6x1]
)
fwhm_laue = 1

fwhm_to_sigma = ct.fwhm_to_sigma  # 0.42466090014400953

# =============================================================================
# %%
# =============================================================================


@numba.njit(nogil=True, cache=False, parallel=True)
def gaussian_dist(x, cen, fwhm):
    sigma = fwhm_to_sigma*np.radians(fwhm)
    return np.exp(-0.5*(x - cen)**2 / sigma**2)


def generate_voxel_coordinates(voxel_size, layer_thickness,
                               ph_radius, ph_thickness):
    """
    Generate voxel coordiates for a specific layer and pinhole definition.

    Parameters
    ----------
    voxel_size : scalar
        The voxel dimension in mm.
    layer_thickness : scalar
        The layer thickness in mm.
    ph_radius : scalar
        The pinhole radius in mm.
    ph_thickness : scalar
        the pinhole thickness in mm.

    Returns
    -------
    numpy.ndarray
        The (n, 3) list of voxel coordinates in the pinhole reference frame
        with the critical radius applied as a function of layer stantdoff.

    """
    # generate voxel coordinates, mask, and flatten them
    voxel_generator_z = gridutil.make_tolerance_grid(
        voxel_size, layer_thickness, 1, adjust_window=False
    )[1]
    z_centers = np.average(
        np.vstack([voxel_generator_z[:-1], voxel_generator_z[1:]]),
        axis=0
    ) + layer_center
    print("will do %d layers" % len(z_centers))
    print(z_centers)

    vcrds = []
    for z_center in z_centers:
        # generate voxel coordinates in within critial radius
        rho_crit = phutil.compute_critical_voxel_radius(
            z_center, ph_radius, ph_thickness
        )
        print("critical radius at z=%f is %f" % (z_center, rho_crit))
        voxel_generator_xy = gridutil.make_tolerance_grid(
            voxel_size, 2*rho_crit, 1, adjust_window=True
        )[1]
        xy_centers = np.average(
            np.vstack([voxel_generator_xy[:-1], voxel_generator_xy[1:]]),
            axis=0
        )
        vx, vy = np.meshgrid(xy_centers, xy_centers)
        rhoc_mask = np.sum(
            np.stack([vx**2 + vy**2], axis=0), axis=0
        ) <= rho_crit**2
        vx = vx[rhoc_mask].flatten()
        vy = vy[rhoc_mask].flatten()

        vcrds.append(
            np.vstack([vx, vy, np.ones_like(vx)*z_center]).T
        )
    return np.vstack(vcrds)


# =============================================================================
# %% OBJECT INSTANTIATION
# =============================================================================


# load instrument and grab the detecor (one for now)
instr = instrument.HEDMInstrument(
    h5py.File(os.path.join(resources_path, ref_config), 'r')
)
det = instr.detectors[det_key]  # !!! only one
bhat = np.atleast_2d(instr.beam_vector)

# layer_center
layer_center = layer_standoff + 0.5*layer_thickness

# also need the reference pixel angles as computed from the origin
ref_ptth, ref_peta = det.pixel_angles()

# gen voxel coordinates
vcrds = generate_voxel_coordinates(
    voxel_size, layer_thickness, ph_radius, ph_thickness
)

# for concurrency
max_workers = os.cpu_count()  # max(1, os.cpu_count() - 1)
start_stop = distribute_tasks(len(vcrds), max_workers=max_workers)
start_stop_all = (start_stop[0][0], start_stop[-1][-1])

mat_dict = material.load_materials_hdf5(
    os.path.join(resources_path, matl_fname),
    dmin=valunits.valWUnit('dmin', 'length', dmin, 'angstrom'),
    kev=valunits.valWUnit('kev', 'energy', instr.beam_energy, 'keV')
)

matl = mat_dict['Ta']
matl.planeData.exclusions = None

# =============================================================================
# %% GRAND LOOP FUNCTION
# =============================================================================


def grand_loop(start_stop, coords, detector, bhat, rho, pinhole_radius,
               pinhole_thickness, perf_acc=None):
    # coords = coords[:10] # limit...
    cstart, cstop = start_stop
    these_coords = coords[cstart:cstop, :]

    setup_t0 = time.perf_counter_ns()
    # need the cartesian pixel coordinates
    py, px = detector.pixel_coords
    pixel_xys = np.vstack([px.flatten(), py.flatten()]).T

    # loop over voxels to aggregate pixel angles and contributing voxel count
    master_ptth = np.zeros(detector.shape, dtype=float)
    voxel_count = np.zeros(detector.shape, dtype=float)
    image = np.zeros(detector.shape, dtype=float)

    # !!! transposed for use as second arg in np.dot()
    reduced_rmat = np.ascontiguousarray(detector.rmat[:, :2].T)
    setup_t1 = time.perf_counter_ns()

    loop_t0 = time.perf_counter_ns()
    acc_cobv = 0
    acc_phc = 0
    acc_other = 0
    acc_xy2g = 0
    # for iv, coord in enumerate(these_coords):
    for iv, coord in enumerate(tqdm(these_coords)):
        # need new beam vector from curent voxel coordinate
        cobv_t0 = time.perf_counter_ns()
        new_bv = phutil.compute_offset_beam_vector(
            bhat, rho, np.atleast_2d(coord)
        )
        cobv_t1 = time.perf_counter_ns()
        det.bvec = new_bv

        # mask detector pixels
        phc_t0 = time.perf_counter_ns()
        mask = phutil.pinhole_constraint(
            pixel_xys, np.array(coord),
            reduced_rmat, detector.tvec,
            pinhole_radius, pinhole_thickness
        )  # no reshape # .reshape(det.shape)
        phc_t1 = time.perf_counter_ns()

        other_t0 = time.perf_counter_ns()
        if np.any(mask):
            # compute pixel angles that satisfy the pinhole constraint
            # print(
            #     ' '.join(f"it {iv}: mask has {np.sum(mask)} pixels set."
            #              f"{pixel_xys.shape} {mask.shape}")
            # )
            reduced_xys = pixel_xys[mask, :]
            mask = mask.reshape(detector.shape)
            ptth = np.nan*np.ones(detector.shape)
            xy2g_t0 = time.perf_counter_ns()
            angs, _ = xfcapi.detectorXYToGvec(
                reduced_xys, detector.rmat, ct.identity_3x3,
                detector.tvec, ct.zeros_3, np.array(coord),
                beamVec=new_bv)
            acc_xy2g += time.perf_counter_ns() - xy2g_t0
            ptth[mask] = angs[0]
            master_ptth = np.nansum(
                np.stack([master_ptth, ptth], axis=0),
                axis=0
            )
            voxel_count += mask
            # =================================================================
            # ⌄⌄⌄SIMULATION
            # =================================================================
            this_image = np.nan*np.ones(detector.shape)
            matl.planeData.tThMax = np.nanmax(ptth)
            tths = matl.planeData.getTTh()
            mult = matl.planeData.getMultiplicity()
            sfac = matl.planeData.structFact
            sfac = 100*sfac/np.max(sfac)

            for line_params in zip(tths, mult, sfac):
                lpf = (1 + np.cos(line_params[0])**2) \
                    / np.cos(0.5*line_params[0]) \
                    / np.sin(0.5*line_params[0])**2 / 2.0
                scl_fac = lpf*line_params[1]*line_params[2]
                this_image[mask] = scl_fac*gaussian_dist(
                    angs[0], line_params[0], fwhm_powder
                )
                image = np.nansum(
                    np.stack([image, this_image], axis=0),
                    axis=0
                )
            # =================================================================
            # ^^^SIMULATION
            # =================================================================
        other_t1 = time.perf_counter_ns()
        acc_cobv += cobv_t1 - cobv_t0
        acc_phc += phc_t1 - phc_t0
        acc_other += other_t1 - other_t0

    loop_t1 = time.perf_counter_ns()

    if perf_acc is not None:
        perf_acc['gl_setup'] = perf_acc.get('gl_setup', 0) \
            + (setup_t1 - setup_t0)
        perf_acc['gl_loop'] = perf_acc.get('gl_loop', 0) \
            + (loop_t1 - loop_t0)
        perf_acc['gl_computeoffset'] = perf_acc.get('gl_compute_offset', 0) \
            + acc_cobv
        perf_acc['gl_constraint'] = perf_acc.get('gl_constraint', 0) \
            + acc_phc
        perf_acc['gl_other'] = perf_acc.get('gl_other', 0) \
            + acc_other
        perf_acc['gl_xy2g'] = perf_acc.get('gl_xy2g', 0) \
            + acc_xy2g
    return np.stack([master_ptth, voxel_count, image], axis=0)


# =============================================================================
# %% EXECUTE VOXEL LOOP
# =============================================================================
if os.name == 'nt':
    mp_ctx = multiprocessing.get_context("spawn")
else:
    mp_ctx = multiprocessing.get_context("fork")
perf_results = None  # dict()

'''
# SERIAL
start = time.time()
master_ptth, voxel_count = grand_loop(
    start_stop_all,
    vcrds, det, bhat, rho, ph_radius, ph_thickness, perf_acc=perf_results
)
print("Serial execution took %.2f seconds"
      % (time.time() - start))
'''

# parallel execution
kwargs = {
    'coords': vcrds,
    'detector': det,
    'bhat': bhat,
    'rho': rho,
    'pinhole_radius': ph_radius,
    'pinhole_thickness': ph_thickness,
    'perf_acc': perf_results,
}
func = partial(grand_loop, **kwargs)

chunksize = pow(10, int(np.floor(np.log10(len(vcrds)//max_workers))))

start = time.time()
with ProcessPoolExecutor(mp_context=mp_ctx,
                         max_workers=max_workers) as executor:
    results = executor.map(func, start_stop, chunksize=chunksize)
print("Concurrent execution took %.2f seconds"
      % (time.time() - start))

if perf_results is not None:
    for key, val in perf_results.items():
        print(f"{key}: {float(val)/10**9}")

# Concatenate all the results together
cresults = np.concatenate(list(results), axis=0)

# sum interleaved arrays
master_ptth = np.sum(cresults[::3], axis=0)

# apply panel buffer if applicable and make 0 counts nan
voxel_count = np.sum(cresults[1::3], axis=0)
voxel_count[np.logical_or(~det.panel_buffer, voxel_count == 0)] = np.nan

corr = np.array(master_ptth/voxel_count - ref_ptth, dtype=np.float32)
corr_img = np.array(corr, dtype=np.float32)
corr_img[np.isnan(corr_img)] = 0.
io.imsave("correction_field_%s.tif" % det_key, corr_img)

# powder pattern
powder_image = np.sum(cresults[2::3], axis=0)
powder_image = powder_image/np.nanmax(voxel_count)
powder_image += 1e-2
powder_image += 10*np.random.rand(*det.shape)
powder_image[np.isnan(voxel_count)] = 0.0
io.imsave("simulated_Ta_%s.tif" % det_key, powder_image)

# =============================================================================
# %% plotting
# =============================================================================

from hexrd.xrdutil import PolarView

fig, ax = plt.subplots()
mappable = ax.imshow(np.degrees(corr), cmap=plt.cm.gnuplot2)
fig.colorbar(mappable)
fig.suptitle(r"relative $2\theta$ displacements")
plt.show()

fig, ax = plt.subplots()
mappable = ax.imshow(voxel_count, cmap=plt.cm.gnuplot2)
fig.colorbar(mappable)
fig.suptitle(r"voxel histogram")
plt.show()


# This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
# then adds a percent sign.
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"


pv = PolarView((12., 113), instr,
               eta_min=0., eta_max=180.,
               pixel_size=(0.1, 0.1))
intensities = pv.warp_image(
    {det_key: corr},
    do_interpolation=True,
    pad_with_nans=True
)
eta_coordinates, tth_coordinates = pv.angular_grid

fig, ax = plt.subplots()
mappable = ax.imshow(
    np.degrees(intensities),
    cmap=plt.cm.inferno,
    extent=np.degrees(pv.extent)
)
ax.axis('auto')
cbar = fig.colorbar(mappable)
cbar.set_label(r'$2\theta_s-2\theta_n$ [deg]')
fig_title = r' '.join(
    [r'$2\theta_s-2\theta_n$,',
     r'$%.0f\mu\mathrm{m}$ voxels,' % (1e3*voxel_size),
     r'$%.0f\mu\mathrm{m}$ standoff' % (1e3*layer_standoff)]
)
fig.suptitle(fig_title)
ax.set_xlim(10, 110)
ax.set_xlabel(r'nominal Bragg angle, $2\theta_n$ [deg]')
ax.set_ylabel(r'azimuth, $\eta_n$ [deg]')
CS = ax.contour(np.degrees(tth_coordinates),
                np.degrees(eta_coordinates),
                np.degrees(intensities), colors='w',
                levels=None)  # [0.2, 0.3, 0.5, 0.8, 1.1])
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)
ax.grid(True)
