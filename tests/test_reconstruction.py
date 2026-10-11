"""Tests of the point cloud from an AR recording (#48) on synthetic recordings of a textured room with known walls."""

import copy

import numpy as np
import pytest

from pv_yield_estimator.core.ar_recording import ar_to_earth
from pv_yield_estimator.core.reconstruction import FramePoses, ReconstructedPoints, camera_matrix, frame_turn_rates, reconstruct_points, select_frame_pairs
from pv_yield_estimator.plotting.static import reconstruction_figure
from synthetic_recording import SyntheticRoom, recording_from_views, room_recording, synthetic_recording


@pytest.fixture(scope="module")
def room():
    """A synthetic room recording (30 frames, 320 × 240, heading offset 30°) and its reconstruction with the default settings."""
    recording, images, scene = room_recording(heading_offset_deg=30.0)
    return recording, images, scene, reconstruct_points(recording, images)


def test_frame_poses_project_like_camera_views():
    """Points projected with the poses from the recording land where the true camera views (earth coordinates) put them, also with an off-centre principal point; pixel centres at whole numbers as OpenCV's."""
    recording, _, views = synthetic_recording(heading_offset_deg=70.0, with_images=False)
    recording["frames"][0]["projection_matrix"][8] = 0.1                 ### principal point shifted left by 5 % of the width
    poses = FramePoses(recording, [(frame["width"], frame["height"]) for frame in recording["frames"]])
    to_earth = ar_to_earth(70.0)
    centres_earth = poses.centres @ to_earth.T
    targets_earth = centres_earth + 5 * np.array([view.forward + 0.2 * view.right - 0.1 * view.up for view in views])
    pixels, depth = poses.project(np.arange(len(views)), targets_earth @ to_earth)
    expected = np.array([view.project(target - centre)[0][0] for view, target, centre in zip(views, targets_earth, centres_earth)])
    expected[0, 0] -= 0.1 * views[0].width / 2
    expected -= 0.5                                                    ### OpenCV's pixel centres at whole numbers
    assert pixels == pytest.approx(expected, abs=1e-6) and np.all(depth > 0)
    rays = poses.rays(np.arange(len(views)), pixels) @ to_earth.T
    assert np.einsum("ni,ni->n", rays, (targets_earth - centres_earth) / np.linalg.norm(targets_earth - centres_earth, axis=1, keepdims=True)) == pytest.approx(1.0)
    assert camera_matrix([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 100, 50) == pytest.approx(np.array([[100, 0, 50], [0, 50, 25], [0, 0, 1]]))


def test_room_reconstruction(room):
    """The points lie on the room's walls (median off by a few cm), give the room's size, and their estimated error is a fair bound."""
    recording, _, scene, points = room
    assert len(points.table) > 2000
    distances = scene.distance_to_walls(points.points)
    assert np.median(distances) < 0.05 and np.percentile(distances, 90) < 0.25
    ### The walls at their known positions: each point assigned to the nearest wall plane; the floor is hardly seen.
    for axis, name in enumerate(("east", "north", "up")):
        for bound in (scene.low[axis], scene.high[axis]):
            on_wall = np.abs(points.points[:, axis] - bound) == np.min(np.abs(np.concatenate([points.points - scene.low, points.points - scene.high], axis=1)), axis=1)
            if np.count_nonzero(on_wall) > 300:
                assert np.median(points.points[on_wall, axis]) == pytest.approx(bound, abs=0.03), (name, bound)
    assert np.mean(distances <= points.table["distance_error_m"].to_numpy()) > 0.8
    assert points.table["n_frames"].min() >= 2 and points.table["triangulation_angle_deg"].min() >= 2.0 and points.table["reprojection_error_px"].max() <= 2.0
    assert points.metadata["heading_source"] == "compass" and points.metadata["heading_offset_deg"] == pytest.approx(30.0, abs=1e-6)
    assert points.metadata["statistics"]["n_points"] == len(points.table) and points.path[-1] == pytest.approx([0, 0.2 * np.sin(4), 0.4], abs=0.02)      ### the adjusted camera path


def test_heading_offset_turns_the_points():
    """The same room recorded with another heading offset gives the same points in earth coordinates; a wrong heading set by hand turns them."""
    recording, images, scene = room_recording(heading_offset_deg=250.0, n_frames=16)
    points = reconstruct_points(recording, images)
    assert np.median(scene.distance_to_walls(points.points)) < 0.05
    turned = reconstruct_points(recording, images, heading_offset_deg=340.0)
    assert turned.metadata["heading_source"] == "manual" and np.median(scene.distance_to_walls(turned.points)) > 2 * np.median(scene.distance_to_walls(points.points))


def test_pure_rotation_gives_no_points():
    """Turning on the spot (1 mm jitter) gives no baseline: no frame pairs, and without the baseline limit the rays meet only at the camera, too close to count."""
    recording, images, _ = room_recording(n_frames=12, moving=False)
    assert len(reconstruct_points(recording, images).table) == 0
    assert len(reconstruct_points(recording, images, min_baseline_m=0.0).table) == 0


def test_pose_adjustment_repairs_noisy_poses(room):
    """Pose errors as seen on a phone (0.5°, 2 cm) shift the points by decimetres; refining the poses from the images brings them back to a few centimetres."""
    from scipy.spatial.transform import Rotation
    recording, images, scene, exact = room
    noisy = copy.deepcopy(recording)
    generator = np.random.default_rng(3)
    for frame in noisy["frames"]:
        frame["position_m"] = (np.array(frame["position_m"]) + generator.normal(0, 0.02, 3)).tolist()
        frame["orientation_xyzw"] = (Rotation.from_rotvec(generator.normal(0, np.radians(0.5) / np.sqrt(3), 3)) * Rotation.from_quat(frame["orientation_xyzw"])).as_quat().tolist()
    unadjusted = reconstruct_points(noisy, images, adjust_poses=False)
    adjusted = reconstruct_points(noisy, images)
    error_unadjusted, error_adjusted = (np.median(scene.distance_to_walls(points.points)) for points in (unadjusted, adjusted))
    assert error_unadjusted > 0.1 and error_adjusted < 0.08 and len(adjusted.table) > len(unadjusted.table)
    statistics = adjusted.metadata["statistics"]
    assert statistics["adjusted_frames"] == 30 and statistics["rotation_correction_deg_median"] > 0.2 and statistics["reprojection_px_median_after_adjustment"] < 1.0


def test_fast_turning_frames_are_not_used(room):
    """Frames taken while turning fast (blurred) are skipped: by the recorded turn rate, else estimated from the neighbouring frames."""
    recording, images, _, _ = room
    rates = frame_turn_rates(recording)
    assert rates == pytest.approx(10.0, rel=0.05)
    fast = copy.deepcopy(recording)
    for frame in fast["frames"][:10]:
        frame["turn_rate_deg_s"] = 40.0
    for frame in fast["frames"][20:]:
        frame["time_s"] = fast["frames"][19]["time_s"] + 0.1 * (fast["frames"].index(frame) - 19)
    ### Frame 19 too: the mean of its slow step before and its fast step after.
    assert np.count_nonzero(frame_turn_rates(fast) > 20) == 21
    assert reconstruct_points(fast, images).metadata["statistics"]["n_frames_used"] == 9


def test_frames_with_lost_tracking_are_not_used():
    """Frames with an emulated position (tracking lost) take no part."""
    recording, images, _ = room_recording(n_frames=12)
    for frame in recording["frames"][::2]:
        frame["emulated_position"] = True
    points = reconstruct_points(recording, images)
    assert points.metadata["statistics"]["n_frames_used"] == 6
    with pytest.raises(ValueError, match="12 frames but 11 images"):
        reconstruct_points(recording, images[:-1])


def test_frame_pairs():
    """Pairs need the baseline and overlapping views; each frame keeps at most the given number of partners, spread over the baselines."""
    views_recording, _, _ = synthetic_recording(n_frames=12, with_images=False)
    poses = FramePoses(views_recording, [(160, 120)] * 12)
    pairs = select_frame_pairs(poses, np.ones(12, dtype=bool), fov_deg=60.0, min_baseline_m=0.2, max_pairs_per_frame=3)
    baselines = np.linalg.norm(poses.centres[pairs[:, 0]] - poses.centres[pairs[:, 1]], axis=1)
    angles = np.degrees(np.arccos(np.einsum("ni,ni->n", poses.forward[pairs[:, 0]], poses.forward[pairs[:, 1]])))
    assert len(pairs) and np.all(baselines >= 0.2) and np.all(angles <= 30.0) and np.all(pairs[:, 0] < pairs[:, 1])
    assert np.bincount(pairs[:, 0]).max() <= 3


def test_points_csv_round_trip_and_plot(room):
    """The CSV keeps the points and the metadata; another kind or a newer format version is refused; the check plot draws."""
    _, _, _, points = room
    text = points.to_csv_text()
    assert text.startswith("# format_version: 1\n# kind: reconstructed_points\n")
    loaded = ReconstructedPoints.from_csv_text(text, "reconstructed_points.csv")
    assert loaded.points == pytest.approx(points.points, abs=1e-4) and loaded.metadata["statistics"] == points.metadata["statistics"]
    with pytest.raises(ValueError, match="expected 'reconstructed_points'"):
        ReconstructedPoints.from_csv_text(text.replace("kind: reconstructed_points", "kind: photo_set"))
    with pytest.raises(ValueError, match="format version 2"):
        ReconstructedPoints.from_csv_text(text.replace("format_version: 1", "format_version: 2"), "points.csv")
    figure = reconstruction_figure(points)
    assert len(figure.axes) == 3 and "points" in figure.axes[0].get_title()


def test_room_scene():
    """The synthetic room: rays hit the walls at the right distances; the texture varies."""
    scene = SyntheticRoom()
    points, axis, side = scene.hit(np.zeros(3), np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]))
    assert points[:3] == pytest.approx(np.array([[4.0, 0, 0], [0, -2.0, 0], [0, 0, 1.2]]), abs=1e-9) and list(axis[:3]) == [0, 1, 2] and list(side[:3]) == [1, 0, 1]
    assert points[3] == pytest.approx([3.5, 3.5, 0], abs=1e-9)
    assert np.std(scene.grey(*scene.hit(np.zeros(3), np.random.default_rng(0).normal(size=(500, 3))))) > 20
    recording = recording_from_views([], [])
    assert recording["frames"] == []
