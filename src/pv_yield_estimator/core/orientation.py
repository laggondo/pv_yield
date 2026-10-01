"""Panel orientation comparison and optimization from the annual sums of the irradiation per sky patch.

The annual direct radiation on a panel with normal n is approximated by Σ_patches max(V_p · n, 0) with V_p the
patch's direct vectors summed over the year. This equals the exact hourly sum Σ max(V · n, 0) except for patches the
panel's plane cuts through, where some sun positions lie in front of the panel and some behind it; the difference in
annual radiation is well below 1 %. Diffuse radiation is exact (isotropic sky). So a fine grid of orientations costs
one matrix product instead of one pass over the hourly data per orientation.
"""

import logging

import numpy as np

from pv_yield_estimator.core.sky import panel_normal

log = logging.getLogger(__name__)

SHADING_CASES = ("unobstructed", "obstructed")


class OrientationGrid:
    """Annual radiation on the panel (kWh/m²) over a grid of tilt and azimuth angles, unobstructed and obstructed.

    `radiation[case]` has shape (len(tilts_deg), len(azimuths_deg)).
    """

    def __init__(self, tilts_deg, azimuths_deg, radiation):
        self.tilts_deg = np.asarray(tilts_deg, dtype=float)
        self.azimuths_deg = np.asarray(azimuths_deg, dtype=float)
        self.radiation = radiation

    def best(self, case="obstructed"):
        """Orientation with the most annual radiation: dict with tilt_deg, azimuth_deg and radiation_kwh_m2."""
        tilt_index, azimuth_index = np.unravel_index(np.argmax(self.radiation[case]), self.radiation[case].shape)
        return {"tilt_deg": float(self.tilts_deg[tilt_index]), "azimuth_deg": float(self.azimuths_deg[azimuth_index]), "radiation_kwh_m2": float(self.radiation[case][tilt_index, azimuth_index])}

    def to_dict(self):
        """Plain lists for JSON export."""
        return {"tilts_deg": self.tilts_deg.tolist(), "azimuths_deg": self.azimuths_deg.tolist(), **{f"radiation_{case}_kwh_m2": np.round(values, 3).tolist() for case, values in self.radiation.items()},
                **{f"best_{case}": self.best(case) for case in SHADING_CASES}}


def grid_angles(fixed_deg, step_deg, stop_deg):
    """Grid angles from 0 to `stop_deg` (inclusive) in steps of `step_deg`, or only the fixed angle if given."""
    if fixed_deg is not None:
        return np.array([float(fixed_deg)])
    if step_deg <= 0:
        raise ValueError(f"Orientation grid steps must be positive, got {step_deg}")
    return np.linspace(0.0, stop_deg, int(round(stop_deg / step_deg)) + 1)


def orientation_grid(irradiation, obstructed_sky, tilt_step_deg=1.0, azimuth_step_deg=2.0, fixed_tilt_deg=None, fixed_azimuth_deg=None, chunk_size=4000, **kwargs):
    """Annual radiation on the panel over tilts 0–90° and azimuths 0–360° (both ends included, for plotting); a fixed angle restricts the grid to it."""
    if not irradiation.sky.same_as(obstructed_sky.sky):
        raise ValueError(f"Irradiation per sky patch ({irradiation.sky.n_patches} patches) and obstructed sky description ({obstructed_sky.sky.n_patches} patches) use different sky discretizations")
    tilts, azimuths = grid_angles(fixed_tilt_deg, tilt_step_deg, 90.0), grid_angles(fixed_azimuth_deg, azimuth_step_deg, 360.0)
    normals = panel_normal(*np.meshgrid(tilts, azimuths, indexing="ij")).reshape(-1, 3)
    free = ~obstructed_sky.obstructed
    ### Only patches the sun passes through contribute direct radiation.
    direct_vectors = irradiation.annual_direct_vectors()
    sunny = np.any(direct_vectors != 0.0, axis=1)
    direct_vectors, sunny_free = direct_vectors[sunny], free[sunny]
    direction_integrals = irradiation.sky.direction_integrals()
    annual_dhi = irradiation.dhi.sum()
    radiation = {case: np.empty(len(normals)) for case in SHADING_CASES}
    for start in range(0, len(normals), chunk_size):
        chunk = normals[start:start + chunk_size].T
        direct = np.maximum(direct_vectors @ chunk, 0.0)
        diffuse = np.maximum(direction_integrals @ chunk, 0.0) * annual_dhi / np.pi
        radiation["unobstructed"][start:start + chunk_size] = direct.sum(axis=0) + diffuse.sum(axis=0)
        radiation["obstructed"][start:start + chunk_size] = direct[sunny_free].sum(axis=0) + diffuse[free].sum(axis=0)
    grid = OrientationGrid(tilts, azimuths, {case: values.reshape(len(tilts), len(azimuths)) / 1000.0 for case, values in radiation.items()})
    log.debug(f"Orientation grid: {len(tilts)} tilts × {len(azimuths)} azimuths; best obstructed {grid.best('obstructed')}")
    return grid
