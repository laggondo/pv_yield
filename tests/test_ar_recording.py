"""Tests of AR video recordings (#47): format check, AR poses as camera views, heading offset from the compass."""

import numpy as np
import pytest

from pv_yield_estimator.core.ar_recording import (ar_to_earth, camera_path, check_recording, covered_fraction, fov_from_projection_matrix, frame_camera_views, heading_offset_from_compass,
                                                  recording_summary, rotation_from_quaternion)
from synthetic_recording import projection_matrix, synthetic_recording


def test_ar_to_earth_and_quaternions():
    """The AR -z axis points to the heading offset's azimuth, y up; quaternions give rotation matrices (identity, and 90° about y turns -z to -x)."""
    assert ar_to_earth(0) @ [0, 0, -1] == pytest.approx([0, 1, 0]) and ar_to_earth(0) @ [1, 0, 0] == pytest.approx([1, 0, 0])
    assert ar_to_earth(90) @ [0, 0, -1] == pytest.approx([1, 0, 0]) and ar_to_earth(90) @ [0, 1, 0] == pytest.approx([0, 0, 1])
    assert ar_to_earth(200) @ [0, 0, -1] == pytest.approx([np.sin(np.radians(200)), np.cos(np.radians(200)), 0])
    assert rotation_from_quaternion([0, 0, 0, 1])[0] == pytest.approx(np.eye(3))
    half = np.sin(np.radians(45))
    assert rotation_from_quaternion([0, half, 0, half])[0] @ [0, 0, -1] == pytest.approx([-1, 0, 0])


def test_fov_from_projection_matrix():
    """The field of view across the longer side, in landscape and portrait."""
    assert fov_from_projection_matrix(projection_matrix(65, 640, 480), 640, 480) == pytest.approx(65)
    assert fov_from_projection_matrix(projection_matrix(65, 480, 640), 480, 640) == pytest.approx(65)


@pytest.mark.parametrize("heading_offset_deg", [0.0, 30.0, 250.0])
def test_frames_as_camera_views(heading_offset_deg):
    """With the true heading offset, the frames' camera views equal the views they were generated from; the camera path is in metres from the reference, east/north/up."""
    recording, _, true_views = synthetic_recording(heading_offset_deg, with_images=False)
    for view, true_view in zip(frame_camera_views(recording, heading_offset_deg), true_views):
        assert (view.azimuth_deg, view.elevation_deg, view.roll_deg, view.fov_deg) == pytest.approx((true_view.azimuth_deg, true_view.elevation_deg, true_view.roll_deg, true_view.fov_deg), abs=1e-6)
        assert (view.width, view.height) == (true_view.width, true_view.height)
    path = camera_path(recording, heading_offset_deg)
    assert path[0] == pytest.approx([0, 0, 0], abs=1e-9) and path[-1] == pytest.approx([2.0, 0.1 * np.sin(3), 0.3])
    ### A wrong heading offset turns all views by the difference.
    turned = frame_camera_views(recording, heading_offset_deg + 10)
    assert (turned[0].azimuth_deg - true_views[0].azimuth_deg) % 360 == pytest.approx(10)


def test_heading_offset_from_compass():
    """The heading offset is estimated from the frames' compass readings, else from the readings just before and after the session; robust against a few wrong readings."""
    recording, _, _ = synthetic_recording(123.0, with_images=False)
    estimate = heading_offset_from_compass(recording)
    assert estimate["heading_offset_deg"] == pytest.approx(123.0, abs=1e-6) and estimate["n_readings"] == 12 and estimate["source"] == "compass"
    for frame in recording["frames"][:3]:
        frame["compass"]["alpha_deg"] += 40                   ### a compass pulled off by metal nearby
    assert heading_offset_from_compass(recording)["heading_offset_deg"] == pytest.approx(123.0, abs=1.0)
    recording["frames"][5]["time_s"] += 2                     ### a stale reading doesn't count
    assert heading_offset_from_compass(recording)["n_readings"] == 11
    before_after, _, _ = synthetic_recording(300.0, compass="before_after", with_images=False)
    estimate = heading_offset_from_compass(before_after)
    assert estimate["heading_offset_deg"] == pytest.approx(300.0, abs=1e-6) and estimate["n_readings"] == 2 and estimate["source"] == "compass_before_after"
    none, _, _ = synthetic_recording(compass="none", with_images=False)
    assert heading_offset_from_compass(none) == {"heading_offset_deg": None, "n_readings": 0, "spread_deg": None, "source": "none"}


def test_recording_summary():
    """Key figures: frames, baseline, path length, the heading (estimated, or set by hand), the covered share of the sky and the GPS position's distance from the site."""
    recording, _, _ = synthetic_recording(45.0, with_images=False)
    recording["frames"][3]["emulated_position"] = True
    summary = recording_summary(recording, site_latitude=47.996, site_longitude=7.859)
    assert summary["n_frames"] == 12 and summary["n_tracking_lost"] == 1 and summary["duration_s"] == pytest.approx(4.4)
    assert summary["heading_offset_deg"] == pytest.approx(45.0) and summary["heading_source"] == "compass" and summary["start_to_end_m"] == pytest.approx(np.linalg.norm([2.0, 0.1 * np.sin(3), 0.3]))
    assert summary["baseline_m"] >= summary["start_to_end_m"] and summary["path_length_m"] >= summary["start_to_end_m"]
    assert 0.05 < summary["covered_fraction"] < 0.6 and summary["gps_distance_km"] == pytest.approx(0.745, abs=0.01)
    assert summary["projection"]["principal_point_offset_px"] == pytest.approx([0, 0]) and summary["projection"]["projection_aspect"] == pytest.approx(160 / 120)
    assert len(summary["views"]) == len(summary["path"]) == 12
    manual = recording_summary(recording, heading_offset_deg=50.0)
    assert manual["heading_source"] == "manual" and manual["views"][0]["azimuth_deg"] == pytest.approx(summary["views"][0]["azimuth_deg"] + 5)
    recording.update(heading_offset_deg=60.0, heading_source="manual")
    assert recording_summary(recording)["heading_offset_deg"] == pytest.approx(60.0)          ### stored when set by hand


def test_covered_fraction():
    """A camera looking straight up with a 90° field of view across a square image sees a known share of the hemisphere."""
    view = frame_camera_views(synthetic_recording(n_frames=1, with_images=False)[0], 0)[0]
    view.__init__(azimuth_deg=0, elevation_deg=90, fov_deg=90, width=100, height=100)
    ### Solid angle of a square pyramid with half angles of 45°: 4 arcsin(sin² 45°) = 2π/3, a third of the hemisphere.
    assert covered_fraction([view]) == pytest.approx(1 / 3, abs=0.005)


def test_check_recording():
    """Format version, kind and frame entries are checked with helpful messages."""
    recording, _, _ = synthetic_recording(with_images=False)
    assert check_recording(recording) is recording
    with pytest.raises(ValueError, match="format version 2"):
        check_recording({**recording, "format_version": 2}, "recording.json")
    with pytest.raises(ValueError, match="expected 'ar_recording'"):
        check_recording({**recording, "kind": "photo_set"})
    broken = {**recording, "frames": [{key: value for key, value in recording["frames"][0].items() if key != "orientation_xyzw"}]}
    with pytest.raises(ValueError, match="frame 1 \\(recording/frame_1.jpg\\) lacks orientation_xyzw"):
        check_recording(broken, "recording.json")
