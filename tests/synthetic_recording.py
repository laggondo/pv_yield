"""Synthetic AR recordings for the tests (#47, #48): known camera poses, consistent compass readings and rendered frames.

Real recordings show the user's home and stay private, so the tests use generated ones. Two scenes:
- `synthetic_recording`: a scene infinitely far away whose colour depends on the direction only (sectors of azimuth
  and bands of elevation), so frames taken from different positions agree where they overlap, and their colours on
  the sky map can be checked.
- `room_recording`: the textured inside of a box-shaped room at known distances, for the point cloud (#48).
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
    views, positions = [], []
    for index in range(n_frames):
        fraction = index / max(n_frames - 1, 1)
        views.append(CameraView(azimuth_deg=135 + 90 * fraction, elevation_deg=10 + 30 * fraction, roll_deg=5 * np.sin(4 * fraction), fov_deg=fov_deg, width=width, height=height))
        positions.append([2.0 * fraction, 0.1 * np.sin(3 * fraction), 0.3 * fraction])
    images = [render_frame(view) for view in views] if with_images else []
    return recording_from_views(views, positions, heading_offset_deg, compass), images, views


def recording_from_views(views, positions_earth, heading_offset_deg=30.0, compass="frames", time_step_s=0.4):
    """A recording dict with frames at the given camera views (earth coordinates) and positions (east, north, up in metres from the reference position), `time_step_s` apart, with poses in AR coordinates for the heading offset.

    `compass` as in `synthetic_recording`.
    """
    to_earth = ar_to_earth(heading_offset_deg)
    frames = []
    for index, (view, position_earth) in enumerate(zip(views, positions_earth)):
        width, height = int(view.width), int(view.height)
        ### Camera to earth: columns image right, image up, backward; the device (portrait) has the same axes as the camera.
        earth_rotation = np.stack([view.right, view.up, -view.forward], axis=1)
        ar_rotation = to_earth.T @ earth_rotation
        alpha, beta, gamma = Rotation.from_matrix(earth_rotation).as_euler("ZXY", degrees=True)
        frame = {"file": frame_file_name(index + 1), "time_s": 0.5 + time_step_s * index, "position_m": (to_earth.T @ np.asarray(position_earth, dtype=float)).tolist(), "orientation_xyzw": Rotation.from_matrix(ar_rotation).as_quat().tolist(),
                 "emulated_position": False, "width": width, "height": height, "projection_matrix": projection_matrix(view.fov_deg, width, height), "screen_angle_deg": 0,
                 "compass": {"alpha_deg": alpha, "beta_deg": beta, "gamma_deg": gamma, "time_s": 0.5 + time_step_s * index - 0.05} if compass == "frames" else None}
        frames.append(frame)
    width, height = (int(views[0].width), int(views[0].height)) if views else (0, 0)
    details = {"created": "2026-10-10T12:00:00Z", "user_agent": "synthetic", "reference_note": "synthetic reference", "gps": {"latitude": 47.996, "longitude": 7.849, "altitude_m": 280.0, "accuracy_m": 5.0},
               "heading_offset_deg": None, "heading_source": None, "diagnostics": {"camera_image": f"{width} x {height}"}}
    if compass == "before_after":
        first, last = frames[0], frames[-1]
        for key, frame, time_shift in (("compass_before", first, -0.3), ("compass_after", last, 0.4)):
            rotation = to_earth @ Rotation.from_quat(frame["orientation_xyzw"]).as_matrix()
            alpha, beta, gamma = Rotation.from_matrix(rotation).as_euler("ZXY", degrees=True)
            details[key] = {"alpha_deg": alpha, "beta_deg": beta, "gamma_deg": gamma, "time_s": frame["time_s"] + time_shift}
    return new_recording(frames, **details)


class SyntheticRoom:
    """An indoor scene for the point cloud tests (#48): the inside of a box-shaped room (earth coordinates, metres from the reference position) whose walls, floor and ceiling carry a random texture.

    The texture is the sum of random grey values on square grids of several cell sizes, interpolated bilinearly, so it
    has corners at all scales for the feature detector. Rendering casts a ray per pixel to the wall it hits.
    """

    def __init__(self, low=(-3.0, -2.0, -1.3), high=(4.0, 3.5, 1.2), cell_sizes_m=(0.03, 0.1, 0.4), seed=1):
        self.low, self.high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        self.cell_sizes_m = cell_sizes_m
        generator = np.random.default_rng(seed)
        ### Per wall (axis, side) and cell size a grid of grey values covering the wall's extent.
        self.grids = {}
        for axis in range(3):
            others = [other for other in range(3) if other != axis]
            for side in range(2):
                extent = self.high[others] - self.low[others]
                self.grids[axis, side] = [generator.uniform(-1, 1, size=(int(extent[0] / cell) + 2, int(extent[1] / cell) + 2)) for cell in cell_sizes_m]

    def hit(self, origin, directions):
        """Points where rays from `origin` (3) in `directions` (n, 3) hit the room's inside, and the axis and side (0 low, 1 high) of the wall hit."""
        directions = np.where(np.abs(directions) < 1e-12, 1e-12, directions)
        bounds = np.where(directions > 0, self.high, self.low)
        distances = (bounds - origin) / directions
        axis = np.argmin(distances, axis=1)
        points = origin + distances[np.arange(len(directions)), axis][:, None] * directions
        return points, axis, (directions[np.arange(len(directions)), axis] > 0).astype(int)

    def grey(self, points, axis, side):
        """Grey value (0–255) of the walls' texture at points on the walls."""
        from scipy.ndimage import map_coordinates
        values = np.full(len(points), 128.0)
        for (wall_axis, wall_side), grids in self.grids.items():
            on_wall = (axis == wall_axis) & (side == wall_side)
            others = [other for other in range(3) if other != wall_axis]
            local = points[on_wall][:, others] - self.low[others]
            for grid, cell, weight in zip(grids, self.cell_sizes_m, (60.0, 40.0, 30.0)):
                values[on_wall] += weight * map_coordinates(grid, (local / cell).T, order=1, mode="nearest")
        return np.clip(values, 0, 255)

    def render(self, view, position):
        """JPEG bytes of the room seen by a camera view (earth coordinates) at a position."""
        from PIL import Image
        width, height = int(view.width), int(view.height)
        columns, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        points, axis, side = self.hit(np.asarray(position, dtype=float), view.directions(np.stack([columns.ravel(), rows.ravel()], axis=1)))
        buffer = io.BytesIO()
        Image.fromarray(self.grey(points, axis, side).reshape(height, width).astype(np.uint8)).save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    def distance_to_walls(self, points):
        """Distance of points (n, 3) from the nearest wall plane, metres."""
        return np.min(np.abs(np.concatenate([points - self.low, points - self.high], axis=1)), axis=1)


def room_recording(heading_offset_deg=30.0, n_frames=30, fov_deg=60.0, width=320, height=240, moving=True, room=None):
    """A recording in a `SyntheticRoom`: two passes of 2 m along the room (east, then back west 0.4 m higher), looking around slowly; without `moving`, the camera turns on the spot (jitter of 1 mm, no baseline).

    Returns (recording dict, list of JPEG bytes, the room).
    """
    room = room or SyntheticRoom()
    jitter = np.random.default_rng(5).normal(0, 0.001, (n_frames, 3))
    views, positions = [], []
    for index in range(n_frames):
        fraction = index / max(n_frames - 1, 1)
        back = fraction > 0.5
        along = 2 * fraction if not back else 2 * (1 - fraction)
        views.append(CameraView(azimuth_deg=(330 + 300 * fraction) % 360, elevation_deg=-10 + 50 * fraction, roll_deg=8 * np.sin(5 * fraction), fov_deg=fov_deg, width=width, height=height))
        positions.append([2.0 * along, 0.2 * np.sin(4 * fraction), 0.4 * back] if moving else jitter[index])
    images = [room.render(view, position) for view, position in zip(views, positions)]
    ### Frames timed so that the camera turns by 10°/s, slow enough not to blur.
    return recording_from_views(views, positions, heading_offset_deg, time_step_s=300 / max(n_frames - 1, 1) / 10), images, room
