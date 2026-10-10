"""Synthetic AR recordings for the tests (#47): known camera poses, consistent compass readings and rendered frames.

Real recordings show the user's home and stay private, so the tests use generated ones. The frames show a scene
infinitely far away whose colour depends on the direction only (sectors of azimuth and bands of elevation), so frames
taken from different positions agree where they overlap, and their colours on the sky map can be checked.
"""

import io

import numpy as np
from scipy.spatial.transform import Rotation

from pv_yield_estimator.core.ar_recording import ar_to_earth, frame_file_name, new_recording
from pv_yield_estimator.core.photo import CameraView
from pv_yield_estimator.core.sky import zenith_azimuth_from_directions

### Colours of the scene: eight azimuth sectors of 45° (from north, clockwise), lighter above 30° elevation.
SECTOR_COLOURS = np.array([[200, 40, 40], [40, 160, 40], [40, 60, 200], [220, 200, 40], [160, 40, 180], [40, 190, 190], [230, 120, 30], [120, 120, 120]], dtype=float)


def scene_colour(directions):
    """RGB colour (0–255) of the synthetic scene seen in the given directions (earth coordinates), shape (n, 3)."""
    zenith, azimuth = zenith_azimuth_from_directions(directions)
    colours = SECTOR_COLOURS[(np.degrees(azimuth) // 45).astype(int) % 8]
    return np.where((zenith < np.radians(60))[:, None], 0.5 * colours + 127, colours)


def projection_matrix(fov_deg, width, height, near=0.1, far=1000.0):
    """WebXR projection matrix (16 entries, column-major) of a centred pinhole camera with the field of view across the image's longer side."""
    half_tan = np.tan(np.radians(fov_deg) / 2)
    tan_x, tan_y = (half_tan, half_tan * height / width) if width >= height else (half_tan * width / height, half_tan)
    return [1 / tan_x, 0, 0, 0, 0, 1 / tan_y, 0, 0, 0, 0, -(far + near) / (far - near), -1, 0, 0, -2 * far * near / (far - near), 0]


def render_frame(view):
    """JPEG bytes of the synthetic scene seen by a camera view."""
    from PIL import Image
    width, height = int(view.width), int(view.height)
    columns, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    colours = scene_colour(view.directions(np.stack([columns.ravel(), rows.ravel()], axis=1)))
    buffer = io.BytesIO()
    Image.fromarray(colours.reshape(height, width, 3).astype(np.uint8)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def synthetic_recording(heading_offset_deg=30.0, n_frames=12, fov_deg=60.0, width=160, height=120, compass="frames", with_images=True):
    """A recording along a 2 m line towards east, turning from south-east to south-west and up; returns (recording dict, list of JPEG bytes, list of the true camera views).

    `compass`: "frames" gives each frame a compass reading (exact device angles of the true orientation), "before_after"
    only the readings just before and after the session, "none" no compass at all.
    """
    to_earth = ar_to_earth(heading_offset_deg)
    frames, images, views = [], [], []
    for index in range(n_frames):
        fraction = index / max(n_frames - 1, 1)
        view = CameraView(azimuth_deg=135 + 90 * fraction, elevation_deg=10 + 30 * fraction, roll_deg=5 * np.sin(4 * fraction), fov_deg=fov_deg, width=width, height=height)
        ### Camera to earth: columns image right, image up, backward; the device (portrait) has the same axes as the camera.
        earth_rotation = np.stack([view.right, view.up, -view.forward], axis=1)
        ar_rotation = to_earth.T @ earth_rotation
        position_earth = np.array([2.0 * fraction, 0.1 * np.sin(3 * fraction), 0.3 * fraction])
        alpha, beta, gamma = Rotation.from_matrix(earth_rotation).as_euler("ZXY", degrees=True)
        frame = {"file": frame_file_name(index + 1), "time_s": 0.5 + 0.4 * index, "position_m": (to_earth.T @ position_earth).tolist(), "orientation_xyzw": Rotation.from_matrix(ar_rotation).as_quat().tolist(),
                 "emulated_position": False, "width": width, "height": height, "projection_matrix": projection_matrix(fov_deg, width, height), "screen_angle_deg": 0,
                 "compass": {"alpha_deg": alpha, "beta_deg": beta, "gamma_deg": gamma, "time_s": 0.5 + 0.4 * index - 0.05} if compass == "frames" else None}
        frames.append(frame)
        views.append(view)
        if with_images:
            images.append(render_frame(view))
    details = {"created": "2026-10-10T12:00:00Z", "user_agent": "synthetic", "reference_note": "synthetic reference", "gps": {"latitude": 47.996, "longitude": 7.849, "altitude_m": 280.0, "accuracy_m": 5.0},
               "heading_offset_deg": None, "heading_source": None, "diagnostics": {"camera_image": f"{width} x {height}"}}
    if compass == "before_after":
        first, last = frames[0], frames[-1]
        for key, frame, time_shift in (("compass_before", first, -0.3), ("compass_after", last, 0.4)):
            rotation = to_earth @ Rotation.from_quat(frame["orientation_xyzw"]).as_matrix()
            alpha, beta, gamma = Rotation.from_matrix(rotation).as_euler("ZXY", degrees=True)
            details[key] = {"alpha_deg": alpha, "beta_deg": beta, "gamma_deg": gamma, "time_s": frame["time_s"] + time_shift}
    return new_recording(frames, **details), images, views
