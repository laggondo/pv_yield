"""Tests of the LiDAR sky obstruction on synthetic point clouds and the sample scan."""

from pathlib import Path

import numpy as np
import pytest

from pv_yield_estimator.core.lidar import LidarSkyObstruction, read_livox_csv, scanner_to_geographic
from pv_yield_estimator.core.obstruction_methods import sky_obstruction_method
from pv_yield_estimator.core.sky import SkyDiscretization, directions_from_zenith_azimuth

DATA = Path(__file__).resolve().parents[1] / "data"

LIVOX_CSV = """Version,LiDAR Index,Rsvd,Error Code,Timestamp Type,Data Type,Timestamp,X,Y,Z,Reflectivity,Tag,Ori_x
47MDN540030520,0.0,0.0,0.0,0.0,0.0,0.0,2,9,12975552
0,0,0,0,0,2,1,0.000000,0.000000,0.000000,0,0,0
0,0,0,0,0,2,1,1.000000,2.000000,3.000000,10,0,0
0,0,0,0,0,2,1,4.000000,5.000000,6.000000,0,0,0
0,0,0,0,0,2,1,-1.000000,0.500000,2.000000,7,0,0
"""


@pytest.fixture(scope="module")
def sky():
    """A medium-resolution discretization."""
    return SkyDiscretization.from_node_count(200)


def test_read_livox_csv_drops_invalid_points():
    """The device line, points without return and points at the origin are dropped."""
    np.testing.assert_array_equal(read_livox_csv(LIVOX_CSV), [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])


def test_scanner_heading_rotation():
    """Scanner x axis points to the heading, y axis 90° counterclockwise (to the left) of it."""
    np.testing.assert_allclose(scanner_to_geographic(np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]), 90.0), [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], atol=1e-15)
    np.testing.assert_allclose(scanner_to_geographic(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 180.0), [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-15)


def cluster(azimuth_deg, elevation_deg, n_points, distance=10.0, spread_deg=0.5, seed=0):
    """Points in the geographic frame clustered around a direction."""
    rng = np.random.default_rng(seed)
    zenith = np.radians(90.0 - elevation_deg + rng.uniform(-spread_deg, spread_deg, n_points))
    azimuth = np.radians(azimuth_deg + rng.uniform(-spread_deg, spread_deg, n_points))
    return distance * directions_from_zenith_azimuth(zenith, azimuth)


def test_synthetic_obstruction_heading_offset_and_threshold(sky):
    """A dense cluster obstructs its patch, a sparse one is filtered as noise; heading and offset move the directions."""
    dense_east, sparse_north = cluster(90.0, 30.0, 50), cluster(0.0, 30.0, 3, seed=1)
    ### Scanner x axis points north, so geographic east is the scanner's -y axis.
    points_scanner = np.concatenate([dense_east, sparse_north]) @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T
    flags = LidarSkyObstruction(scanner_heading_deg=0.0, min_points=10).obstructed_flags(sky, points_scanner)
    east_patch, north_patch = sky.find_patches(directions_from_zenith_azimuth(np.radians([60.0, 60.0]), np.radians([90.0, 0.0])))
    assert flags[east_patch] and not flags[north_patch]
    assert 1 <= flags.sum() <= 4
    ### min_points_percent: 3 of 53 points is 5.7 %, so a threshold of 5 % keeps the sparse cluster.
    assert LidarSkyObstruction(scanner_heading_deg=0.0, min_points_percent=5.0).obstructed_flags(sky, points_scanner)[north_patch]
    ### A panel 20 m above the scanner sees all points below its horizon.
    assert not LidarSkyObstruction(scanner_heading_deg=0.0, panel_offset_m=[0, 0, 20.0]).obstructed_flags(sky, points_scanner).any()


def test_obstructed_sky_metadata_and_method_registry(sky):
    """The config selects the method; its settings end up in the metadata."""
    method = sky_obstruction_method(method="lidar", lidar={"scanner_heading_deg": 10.0, "min_points": 2})
    obstructed_sky = method.obstructed_sky(sky, cluster(180.0, 20.0, 20), metadata={"latitude": 48.0})
    assert obstructed_sky.metadata["method"] == "lidar"
    assert obstructed_sky.metadata["method_settings"]["scanner_heading_deg"] == 10.0
    assert obstructed_sky.metadata["latitude"] == 48.0
    with pytest.raises(ValueError, match="photo"):
        sky_obstruction_method(method="photo")


def test_sample_scan():
    """The sample scan gives the obstructed fraction and sky view factor of the legacy code (see test_legacy_comparison)."""
    method = LidarSkyObstruction(scanner_heading_deg=188.1)
    with open(DATA / "2026-06-01_22-25-29_red_red.csv", encoding="utf-8") as point_cloud_file:
        points = method.read_input(point_cloud_file)
    obstructed_sky = method.obstructed_sky(SkyDiscretization.from_node_count(500), points)
    assert obstructed_sky.obstructed.sum() == 626
    assert obstructed_sky.sky_view_factor() == pytest.approx(0.5500, abs=1e-4)


def test_read_livox_csv_rejects_other_files():
    """Another file, e.g. an obstructed sky description, gets a message naming the expected columns instead of a parser error."""
    with pytest.raises(ValueError, match="Not a Livox CSV point cloud.*X, Y, Z and Reflectivity"):
        read_livox_csv('{"format_version": 1, "kind": "obstructed_sky"}')
