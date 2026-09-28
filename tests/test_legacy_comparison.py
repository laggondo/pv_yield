"""Regression test against the legacy code on the sample data (#23).

The reference figures in `legacy_reference/legacy_key_figures.yaml` come from `legacy_reference/run_legacy.py`, which
runs the unchanged legacy functions with the parameters of the legacy main block. Expected differences:
- Obstruction: none. Same nodes and triangles (for the same node count), and the scanner rotation converts exactly
  (`scanner_heading_deg = 270 + lidar_yaw_deg`), so the obstructed patches are identical.
- Direct radiation (< 1 % unobstructed, < 2 % obstructed): pvlib's sun position (NREL SPA with refraction) instead of
  the legacy approximation; the legacy code evaluates cos(incidence) once at mid-hour and loses the direct radiation
  of sunrise and sunset hours whose midpoint is below the horizon, while the new code evaluates it per sub-step and
  keeps the hour's DNI; sub-steps with the sun below the horizon don't count as unobstructed in the legacy code.
- Diffuse radiation: equal for a horizontal panel (same sky view factor); on tilted panels the legacy code uses the
  horizontal diffuse radiation unchanged, the new code the isotropic view factor (1 + cos tilt) / 2 of the sky and
  weights each patch by its angle to the panel normal.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from pv_yield_estimator.core.irradiation import compute_patch_irradiation
from pv_yield_estimator.core.lidar import LidarSkyObstruction
from pv_yield_estimator.core.panel_yield import YieldEstimator
from pv_yield_estimator.core.sky import SkyDiscretization
from pv_yield_estimator.core.weather import load_weather, resolve_site

REPOSITORY = Path(__file__).resolve().parents[1]
LEGACY = yaml.safe_load((REPOSITORY / "tests" / "legacy_reference" / "legacy_key_figures.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def obstructed_sky():
    """Obstructed sky from the sample scan with the legacy settings."""
    parameters = LEGACY["parameters"]
    method = LidarSkyObstruction(scanner_heading_deg=270.0 + parameters["lidar_yaw_deg"], min_points=parameters["min_points_per_patch"])
    with open(REPOSITORY / "data" / "2026-06-01_22-25-29_red_red.csv", encoding="utf-8") as point_cloud_file:
        points = method.read_input(point_cloud_file)
    return method.obstructed_sky(SkyDiscretization.from_node_count(parameters["n_sky_nodes"]), points)


@pytest.fixture(scope="module")
def irradiation(obstructed_sky):
    """Irradiation per sky patch from the sample weather file."""
    weather = load_weather((REPOSITORY / "data" / "Freiburg-hour.csv").read_text(encoding="utf-8"))
    return compute_patch_irradiation(weather, obstructed_sky.sky, **resolve_site(weather))


def test_obstructed_patches_identical(obstructed_sky):
    """The same patches are obstructed as in the legacy code, and the horizontal sky view factor agrees."""
    sky = obstructed_sky.sky
    legacy_patches = np.array(LEGACY["obstructed_patch_nodes_enu"])
    assert sky.n_patches == LEGACY["n_patches"]
    patch_of_legacy = sky.find_patches(legacy_patches.mean(axis=1))
    np.testing.assert_allclose(sky.nodes[sky.triangles[patch_of_legacy]].sum(axis=1), legacy_patches.sum(axis=1), atol=1e-8)
    assert set(patch_of_legacy) == set(np.nonzero(obstructed_sky.obstructed)[0])
    assert obstructed_sky.sky_view_factor() == pytest.approx(LEGACY["sky_view_factor_horizontal"], abs=1e-5)


@pytest.mark.parametrize("orientation", list(LEGACY["orientations"]))
def test_annual_radiation_close_to_legacy(irradiation, obstructed_sky, orientation):
    """Annual radiation on the panel within the tolerances explained in the module docstring."""
    reference = LEGACY["orientations"][orientation]
    legacy = reference["annual_kwh_m2"]
    tilt_deg = reference["tilt_deg"]
    figures = YieldEstimator(irradiation, obstructed_sky, panel={"tilt_deg": tilt_deg, "azimuth_deg": reference["azimuth_compass_deg"]}).run().key_figures()
    assert figures["annual_direct_unobstructed_kwh_m2"] == pytest.approx(legacy["beam_unobstructed"], rel=0.01)
    assert figures["annual_direct_obstructed_kwh_m2"] == pytest.approx(legacy["beam_obstructed"], rel=0.02)
    sky_fraction = (1 + np.cos(np.radians(tilt_deg))) / 2
    assert figures["annual_diffuse_unobstructed_kwh_m2"] == pytest.approx(legacy["diffuse_unobstructed"] * sky_fraction, rel=2e-3)
    if tilt_deg == 0:
        assert figures["annual_diffuse_obstructed_kwh_m2"] == pytest.approx(legacy["diffuse_obstructed"], rel=1e-4)
        assert figures["shading_loss"] == pytest.approx(legacy["shading_loss"], abs=0.005)
