"""Sky obstruction from a LiDAR point cloud (first version's method)."""

import io
import logging

import numpy as np
import pandas as pd

from pv_yield_estimator.core.sky import SkyObstructionMethod

log = logging.getLogger(__name__)


def read_livox_csv(content):
    """Read a Livox point cloud CSV (as exported by Livox Viewer) and return the valid points, shape (n, 3), in metres.

    `content` is the file content as text or an open file-like object (the core does no file access). The second
    line holds device information and is skipped; points without return (reflectivity 0) or at the origin are
    dropped. Coordinates are in the scanner frame: x forward, y left, z up.
    """
    table = pd.read_csv(io.StringIO(content) if isinstance(content, str) else content, header=0, skiprows=[1], usecols=["X", "Y", "Z", "Reflectivity"])
    points = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    valid = (table["Reflectivity"].to_numpy() > 0) & np.any(points != 0.0, axis=1)
    log.info(f"LiDAR point cloud: {len(points)} points, {np.count_nonzero(valid)} valid")
    return points[valid]


### Point cloud readers by format name; further formats plug in here.
POINT_CLOUD_READERS = {"livox_csv": read_livox_csv}


def read_point_cloud(content, point_cloud_format="livox_csv", **kwargs):
    """Read a point cloud with the reader registered for `point_cloud_format`."""
    if point_cloud_format not in POINT_CLOUD_READERS:
        raise ValueError(f"Unknown point cloud format {point_cloud_format!r}; known: {', '.join(POINT_CLOUD_READERS)}")
    return POINT_CLOUD_READERS[point_cloud_format](content)


def scanner_to_geographic(points, scanner_heading_deg=0.0):
    """Rotate points from the scanner frame (x forward, y left, z up) into x east, y north, z up.

    `scanner_heading_deg` is the compass azimuth (0° = north, clockwise) that the scanner's x axis points to; the
    scanner is assumed to be level. Legacy `lidar_yaw_deg` (clockwise angle from the scanner's y axis to south)
    corresponds to `scanner_heading_deg = 270 + lidar_yaw_deg`.
    """
    heading = np.radians(scanner_heading_deg)
    x_axis, y_axis = np.array([np.sin(heading), np.cos(heading), 0.0]), np.array([-np.cos(heading), np.sin(heading), 0.0])
    return points[:, [0]] * x_axis + points[:, [1]] * y_axis + points[:, [2]] * np.array([0.0, 0.0, 1.0])


class LidarSkyObstruction(SkyObstructionMethod):
    """Obstructed sky from a LiDAR point cloud: a patch is obstructed if it contains enough points.

    Points are rotated into the geographic frame, shifted by the panel offset so that directions are seen from the
    panel, and restricted to the upper hemisphere. The minimum number of points per obstructed patch filters noise;
    it is either an absolute count (`min_points`) or, if `min_points_percent` is set, a percentage of all points above
    the horizon. The sky obstruction is evaluated at a single point per panel (#10).
    """

    method_name = "lidar"

    def __init__(self, scanner_heading_deg=0.0, panel_offset_m=(0.0, 0.0, 0.0), min_points=10, min_points_percent=None, point_cloud_format="livox_csv", **kwargs):
        self.scanner_heading_deg = float(scanner_heading_deg)
        self.panel_offset_m = np.asarray(panel_offset_m, dtype=float)
        if self.panel_offset_m.shape != (3,):
            raise ValueError(f"panel_offset_m must be three numbers [east, north, up] in metres, got {panel_offset_m!r}")
        self.min_points = min_points
        self.min_points_percent = min_points_percent
        self.point_cloud_format = point_cloud_format

    @property
    def settings(self):
        """Settings recorded in the metadata of the obstructed sky description."""
        return {"scanner_heading_deg": self.scanner_heading_deg, "panel_offset_m": self.panel_offset_m.tolist(), "min_points": self.min_points, "min_points_percent": self.min_points_percent, "point_cloud_format": self.point_cloud_format}

    def read_input(self, content):
        """Read the point cloud from file content, in the configured format."""
        return read_point_cloud(content, self.point_cloud_format)

    def directions_from_panel(self, points):
        """Directions (not normalized) from the panel to the points above its horizon, in the geographic frame."""
        directions = scanner_to_geographic(points, self.scanner_heading_deg) - self.panel_offset_m
        return directions[directions[:, 2] > 0.0]

    def obstructed_flags(self, sky, points):
        """Obstructed flag per patch: at least the minimum number of points (directions from the panel) in the patch."""
        directions = self.directions_from_panel(points)
        if self.min_points_percent is not None:
            threshold = self.min_points_percent / 100.0 * len(directions)
        else:
            threshold = self.min_points
        counts = np.bincount(sky.find_patches(directions), minlength=sky.n_patches)
        log.info(f"LiDAR: {len(directions)} points above the panel's horizon, minimum {threshold:g} points per obstructed patch")
        return counts >= max(threshold, 1)
