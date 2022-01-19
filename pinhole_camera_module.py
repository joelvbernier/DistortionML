import numpy as np

import numba

epsf = np.finfo(float).eps
sqrt_epsf = np.sqrt(epsf)


# @numba.njit(nogil=True, cache=False, parallel=False)
@numba.njit
def row_norm(a, out=None):
    n, dim = a.shape
    out = out if out is not None else np.empty((n,))

    for i in range(n):
        sqr_norm = a[i, 0] * a[i, 0]
        for j in range(1, dim):
            sqr_norm += a[i, j]*a[i, j]

        out[i] = np.sqrt(sqr_norm)

    return out


# @numba.njit(parallel=True, nogil=True)
@numba.njit
def row_sqrnorm(a, out=None):
    n, dim = a.shape
    out = out if out is not None else np.empty((n,))

    for i in range(n):
        sqr_norm = a[i, 0] * a[i, 0]
        for j in range(1, dim):
            sqr_norm += a[i, j]*a[i, j]

        out[i] = sqr_norm

    return out


# @numba.njit(nogil=True, cache=False, parallel=False)
def unit_vector(a, out=None):
    out = out if out is not None else np.empty_like(a)

    n, dim = a.shape
    for i in range(n):
        sqr_norm = a[i, 0] * a[i, 0]
        for j in range(1, dim):
            sqr_norm += a[i, j]*a[i, j]

        if sqr_norm > sqrt_epsf:
            recip_norm = 1.0 / np.sqrt(sqr_norm)
            out[i, :] = a[i, :] * recip_norm
        else:
            out[i, :] = a[i, :]

    return out


# @numba.njit(nogil=True, cache=False, parallel=True)
def compute_offset_beam_vector(bv, rho, tv, out=None):
    out = out if out is not None else np.empty_like(bv)
    return -unit_vector(-bv*rho - tv)


# @numba.njit(nogil=True, cache=False, parallel=True)
def ray_plane(rays, rmat, tvec):
    """
    Calculate the primitive ray-plane intersection.

    Parameters
    ----------
    rays : array_like, (n, 3)
        The vstacked collection of rays to intersect with the specified plane.
    rmat : array_like, (3, 3)
        The rotation matrix defining the orientation of the plane in the
        reference CS.  The plane normal is the last column.
    tvec : array_like, (3,)
        The translation vector components in the reference frame used to
        describe a point on the plane (in this case the origin of the local
        planar CS).

    Returns
    -------
    ndarray, (n, 2)
        The local planar (x, y) coordinates of the intersection points, NaN if
        no intersection.

    NOTES
    -----

    """
    nhat = np.ascontiguousarray(rmat[:, 2])
    rays = np.atleast_2d(rays)  # shape (npts, 3)
    output = np.nan*np.ones_like(rays)

    numerator = np.dot(tvec, nhat)
    for i in range(len(rays)):
        denominator = np.dot(rays[i, :], nhat)
        if denominator < 0:
            output[i, :] = rays[i, :] * numerator / denominator
    return np.dot(output - tvec, rmat)[:, :2]


# @numba.njit(parallel=True, nogil=True)
@numba.njit
def ray_plane_trivial(rays, tvec):
    """
    Calculate the primitive ray-plane intersection _without_ rotation

    Parameters
    ----------
    rays : array_like, (n, 3)
        The vstacked collection of rays to intersect with the specified plane.
    tvec : array_like, (3,)
        The translation vector components in the reference frame used to
        describe a point on the plane (in this case the origin of the local
        planar CS).

    Returns
    -------
    ndarray, (n, 2)
        The local planar (x, y) coordinates of the intersection points, NaN if
        no intersection.

    NOTES
    -----

    """
    rays = np.atleast_2d(rays)  # shape (npts, 3)
    nrays = len(rays)
    assert 3 == rays.shape[-1]
    # output = np.nan*np.ones_like(rays)
    output = np.empty((nrays, 2))
    output.fill(np.nan)

    numerator = tvec[2]
    for i in range(len(rays)):
        denominator = rays[i, 2]
        if denominator < 0:
            factor = numerator/denominator
            output[i, 0] = rays[i, 0] * factor - tvec[0]
            output[i, 1] = rays[i, 1] * factor - tvec[1]

    return output


# numba parallel is underwherlming in this case
# @numba.njit(parallel=True, nogil=True)
@numba.njit
def pinhole_constraint_helper(rays, tvecs, radius, result):
    """
    The whole operations of the pinhole constraint put together in
    a single function.

    returns an array with booleans for the rays that pass the constraint
    """
    nrays = len(rays)
    nvecs = len(tvecs)
    sqr_radius = radius*radius
    for ray_index in numba.prange(nrays):
        denominator = rays[ray_index, 2]

        if denominator > 0.:
            result[ray_index] = False
            continue

        is_valid = True
        for tvec_index in range(nvecs):
            numerator = tvecs[tvec_index, 2]
            factor = numerator/denominator
            plane_x = rays[ray_index, 0]*factor - tvecs[tvec_index, 0]
            plane_y = rays[ray_index, 1]*factor - tvecs[tvec_index, 1]
            sqr_norm = plane_x*plane_x + plane_y*plane_y
            if sqr_norm > sqr_radius:
                is_valid = False
                break

        result[ray_index] = is_valid

    return result


# @numba.njit(nogil=True, cache=False, parallel=True)
def pinhole_constraint(pixel_xys, voxel_vec, rmat_d_reduced, tvec_d,
                       radius, thickness):
    """
    Applies pinhole aperature constraint to a collection of rays.

    Parameters
    ----------
    pixel_xys : contiguous ndarray
        (n, 2) array of pixel (x, y) coordinates in the detector frame.
    voxel_vec : contiguous ndarray
        (3, ) vector of voxel COM in the lab frame.
    rmat_d_reduced : contiguous ndarray
        (2, 3) array taken from the (3, 3) detector rotation matrix;
        specifically rmat[:, :2].T
    tvec_d : contiguous ndarray
        (3, ) detector tranlastion vector.
    radius : scalar
        pinhole radius in mm.
    thickness : scalar
        pinhole thickness in mm.

    Returns
    -------
    ndarray, bool
        (n, ) boolean vector where True denotes a pixel that can "see" the
        specified voxel.

    Notes
    -----
    !!! Pinhole plane normal is currently FIXED to [0, 0, 1]
    """
    # '''
    pv_ray_lab = np.dot(pixel_xys, rmat_d_reduced) + (tvec_d - voxel_vec)
    tvecs = np.empty((2, 3))
    tvecs[0] = -voxel_vec
    tvecs[1] = np.r_[0., 0., -thickness] - voxel_vec
    result = np.empty((len(pixel_xys)), dtype=np.bool_)

    return pinhole_constraint_helper(pv_ray_lab, tvecs, radius, result)
    '''
    tvec_ph_b = np.array([0., 0., -thickness])
    pv_ray_lab = np.dot(pixel_xys, rmat_d_reduced) + (tvec_d - voxel_vec)
    # !!! was previously performing unnecessary trival operations
    # rmat_ph = np.eye(3)
    # fint = row_norm(ray_plane(pv_ray_lab, rmat_ph, -voxel_vec))
    # bint = row_norm(ray_plane(pv_ray_lab, rmat_ph, tvec_ph_b - voxel_vec))
    fint = row_sqrnorm(ray_plane_trivial(pv_ray_lab, -voxel_vec))
    bint = row_sqrnorm(ray_plane_trivial(pv_ray_lab, tvec_ph_b - voxel_vec))

    sqr_radius = radius * radius
    return np.logical_and(fint <= sqr_radius, bint <= sqr_radius)
    '''


def compute_critical_voxel_radius(offset, radius, thickness):
    """
    Compute the offset-sepcific critical radius of a pinhole aperture.

    Parameters
    ----------
    offset : scalar
        The offset from the front of the pinhole to the layer position.
    radius : scalar
        pinhole radius.
    thickness : scalar
        pinhole thickness (cylinder height).

    Returns
    -------
    scalar
        the critical _radius_ for a voxel to ray to clear the pinhole aperture.
    """
    return radius*(2*offset/thickness + 1)
