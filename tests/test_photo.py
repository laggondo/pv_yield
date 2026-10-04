"""Tests of the photo method's camera geometry and of manual edits of the obstructed sky description (#17, #12)."""

import numpy as np
import pytest

from pv_yield_estimator.core.photo import CameraView, camera_orientation_from_axes, camera_orientation_from_device
from pv_yield_estimator.core.sky import ObstructedSky, SkyDiscretization, directions_from_zenith_azimuth


def orientation(alpha, beta, gamma, screen_angle=0.0):
    """Camera orientation from device angles, as a tuple (azimuth, elevation, roll) in degrees."""
    result = camera_orientation_from_device(alpha, beta, gamma, screen_angle)
    return result["azimuth_deg"], result["elevation_deg"], result["roll_deg"]


def test_camera_orientation_from_device():
    """Upright phone in portrait: the back camera looks at the horizon in the compass direction 360° − alpha; tilting the top backwards raises the view; landscape and roll follow the screen."""
    assert orientation(0, 90, 0) == pytest.approx((0, 0, 0), abs=1e-9)
    assert orientation(270, 90, 0) == pytest.approx((90, 0, 0), abs=1e-9)                ### alpha counts counterclockwise: facing east
    assert orientation(180, 120, 0) == pytest.approx((180, 30, 0), abs=1e-9)             ### top tilted backwards by 30°: looking 30° up, south
    ### Landscape with the top of the phone to the left (screen angle 90°), top pointing north, screen facing west: the camera looks east, level.
    assert orientation(0, 0, -90, 90) == pytest.approx((90, 0, 0), abs=1e-9)
    ### Upright in portrait, gamma turns the phone about the vertical: looking further west.
    assert orientation(0, 90, 10) == pytest.approx((350, 0, 0), abs=1e-9)
    ### Roll: positive when the image's right edge is raised.
    angle = np.radians(10)
    rolled = camera_orientation_from_axes(right=[np.cos(angle), 0, np.sin(angle)], up=[-np.sin(angle), 0, np.cos(angle)], forward=[0, 1, 0])
    assert (rolled["azimuth_deg"], rolled["elevation_deg"], rolled["roll_deg"]) == pytest.approx((0, 0, 10), abs=1e-9)
    ### Lying flat on its back with the top to the north: the camera looks at the nadir, the image top points north.
    assert orientation(0, 0, 0)[:2] == pytest.approx((0, -90), abs=1e-9)
    ### Tilted backwards over the zenith from facing north: azimuth stays north, the image top then points south.
    assert orientation(0, 180, 0)[:2] == pytest.approx((0, 90), abs=1e-6)


def test_camera_view_projection():
    """The image centre shows the viewing direction; the edge of the longer side lies at half the field of view; pixels and directions are inverse; points behind the camera are flagged."""
    view = CameraView(azimuth_deg=180, elevation_deg=30, roll_deg=0, fov_deg=60, width=800, height=600)
    pixels, in_front = view.project([directions_from_zenith_azimuth(np.radians(60), np.pi), [0.0, 1.0, 0.0]])
    assert pixels[0] == pytest.approx((400, 300)) and list(in_front) == [True, False]
    edge, _ = view.project(np.cos(np.radians(30)) * view.forward - np.sin(np.radians(30)) * view.right)
    assert edge[0] == pytest.approx((0, 300), abs=1e-6)                                  ### half the field of view to the left: left edge of the longer side
    upper, _ = view.project(directions_from_zenith_azimuth(np.radians(50), np.pi))
    assert upper[0, 0] == pytest.approx(400) and upper[0, 1] < 300                       ### higher elevation is up (smaller pixel y)
    random_pixels = np.random.default_rng(0).uniform([0, 0], [800, 600], size=(50, 2))
    assert view.project(view.directions(random_pixels))[0] == pytest.approx(random_pixels)
    rolled = CameraView(azimuth_deg=180, elevation_deg=0, roll_deg=10, fov_deg=60, width=800, height=600)
    west, _ = rolled.project(directions_from_zenith_azimuth(np.pi / 2, np.radians(200)))
    assert west[0, 0] > 400 and west[0, 1] > 300                                         ### right edge raised: the horizon to the right (west when facing south) appears lower
    with pytest.raises(ValueError, match="field of view"):
        CameraView(fov_deg=180)


def test_patch_edges_are_straight_in_photo():
    """Patch edges (great circle arcs) project to straight lines: the midpoint of an edge's arc lies on the line between its projected corners."""
    sky = SkyDiscretization.from_node_count(200)
    view = CameraView(azimuth_deg=200, elevation_deg=40, roll_deg=5, fov_deg=70, width=1000, height=750)
    start, end = sky.nodes[sky.triangles[:, 0]], sky.nodes[sky.triangles[:, 1]]
    middle = start + end
    (start_px, start_front), (end_px, end_front), (middle_px, _) = view.project(start), view.project(end), view.project(middle)
    visible = start_front & end_front
    along, towards_middle = end_px[visible] - start_px[visible], middle_px[visible] - start_px[visible]
    cross = along[:, 0] * towards_middle[:, 1] - along[:, 1] * towards_middle[:, 0]
    assert np.count_nonzero(visible) > 10 and np.abs(cross).max() < 1e-6 * np.abs(start_px[visible]).max() ** 2


def test_obstructed_sky_edited_records_methods():
    """Edits keep the discretization, bake in the flags and record the contributing methods and the editing sessions."""
    sky = SkyDiscretization.from_node_count(50)
    free = ObstructedSky.free(sky)
    assert free.contributing_methods() == []
    flags = np.zeros(sky.n_patches, dtype=bool)
    flags[:5] = True
    photo = free.edited(flags, ["photo"], {"photos": [{"azimuth_deg": 180}]})
    assert photo.metadata["method"] == "photo" and photo.metadata["edits"][0]["n_changed"] == 5 and photo.metadata["edits"][0]["photos"] == [{"azimuth_deg": 180}]
    lidar = ObstructedSky(sky, flags, {"method": "lidar", "input_file": "scan.csv"})
    flags_edited = flags.copy()
    flags_edited[0] = False
    combined = lidar.edited(flags_edited, ["sky_map", "photo"]).edited(flags_edited, ["photo"])
    assert combined.metadata["methods"] == ["lidar", "sky_map", "photo"] and combined.metadata["method"] == "lidar+sky_map+photo"
    assert [edit["n_changed"] for edit in combined.metadata["edits"]] == [1, 0] and combined.metadata["input_file"] == "scan.csv"
    assert ObstructedSky.from_dict(combined.to_dict()).metadata["methods"] == ["lidar", "sky_map", "photo"]
    with pytest.raises(ValueError, match="one entry per patch"):
        lidar.edited(flags[:-1], ["photo"])
