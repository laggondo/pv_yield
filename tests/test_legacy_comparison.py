"""Regression test against the legacy code on the sample LiDAR scan (#23).

The reference figures in `legacy_reference/legacy_key_figures.yaml` come from `legacy_reference/run_legacy.py`, which
ran the unchanged legacy functions with the parameters of the legacy main block. The obstruction is identical: same
nodes and triangles (for the same node count), and the scanner rotation converts exactly
(`scanner_heading_deg = 270 + lidar_yaw_deg`), so the obstructed patches are the same.

The recorded annual radiation values were computed with a MeteoNorm weather file that has since been removed from the
repository (licence), as has `legacy_code/utilityLib.py`, so neither the radiation comparison nor `run_legacy.py`
can be re-run. The comparison is documented in laggondo/pv_yield_estimator#30: direct radiation within 1 %
(unobstructed) and 2 % (obstructed) of the legacy values, diffuse equal for a horizontal panel.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from pv_yield_estimator.core.lidar import LidarSkyObstruction
from pv_yield_estimator.core.sky import SkyDiscretization

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


def test_obstructed_patches_identical(obstructed_sky):
    """The same patches are obstructed as in the legacy code, and the horizontal sky view factor agrees."""
    sky = obstructed_sky.sky
    legacy_patches = np.array(LEGACY["obstructed_patch_nodes_enu"])
    assert sky.n_patches == LEGACY["n_patches"]
    patch_of_legacy = sky.find_patches(legacy_patches.mean(axis=1))
    np.testing.assert_allclose(sky.nodes[sky.triangles[patch_of_legacy]].sum(axis=1), legacy_patches.sum(axis=1), atol=1e-8)
    assert set(patch_of_legacy) == set(np.nonzero(obstructed_sky.obstructed)[0])
    assert obstructed_sky.sky_view_factor() == pytest.approx(LEGACY["sky_view_factor_horizontal"], abs=1e-5)
