"""Photo method of the sky obstruction (#17): camera geometry to overlay the sky patches on a photo.

The user marks obstructed sky patches by hand on photos (or on a map of the sky hemisphere); this module provides the
camera orientation from the phone's sensors and the projection of sky directions to pixels. The photo is assumed to be
taken from the panel position, so no offset is applied (#11).

Coordinate system as in `sky`: x east, y north, z up; azimuths follow the compass (0° = north, clockwise).
"""

import numpy as np

from pv_yield_estimator.core.sky import directions_from_zenith_azimuth

### Viewing directions closer to the zenith or nadir than this (sine of the angle) have no defined azimuth; the image's
### top edge then defines it.
VERTICAL_TOLERANCE = 1e-6


def rotation_from_device_angles(alpha_deg, beta_deg, gamma_deg):
    """Rotation matrix from device to earth coordinates for the W3C device orientation angles (Z-X'-Y'' intrinsic).

    Device axes: x to the right of the screen, y to its top, z out of the screen towards the user. Earth axes: x east,
    y north, z up, which holds for absolute orientation (alpha relative to north, counterclockwise seen from above).
    """
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    rotation_z = np.array([[np.cos(alpha), -np.sin(alpha), 0.0], [np.sin(alpha), np.cos(alpha), 0.0], [0.0, 0.0, 1.0]])
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(beta), -np.sin(beta)], [0.0, np.sin(beta), np.cos(beta)]])
    rotation_y = np.array([[np.cos(gamma), 0.0, np.sin(gamma)], [0.0, 1.0, 0.0], [-np.sin(gamma), 0.0, np.cos(gamma)]])
    return rotation_z @ rotation_x @ rotation_y


def camera_orientation_from_axes(right, up, forward):
    """Viewing azimuth, elevation and roll (degrees) of a camera whose image right, image up and viewing direction are given in earth coordinates.

    Roll is the rotation about the viewing direction, positive when the image's right edge is raised. Looking
    straight up or down, the azimuth is that of the image's top edge (straight up: the direction behind the camera's
    top), so that the result is continuous when tilting the camera over the zenith from a fixed heading.
    """
    right, up, forward = (np.asarray(vector, dtype=float) / np.linalg.norm(vector) for vector in (right, up, forward))
    heading = forward[:2] if np.hypot(*forward[:2]) > VERTICAL_TOLERANCE else -np.sign(forward[2]) * up[:2]
    azimuth_rad = np.arctan2(heading[0], heading[1])
    elevation_rad = np.arcsin(np.clip(forward[2], -1.0, 1.0))
    level_right, level_up = level_camera_axes(azimuth_rad, elevation_rad)
    roll_rad = np.arctan2(right @ level_up, right @ level_right)
    return {"azimuth_deg": float(np.degrees(azimuth_rad) % 360.0), "elevation_deg": float(np.degrees(elevation_rad)), "roll_deg": float(np.degrees(roll_rad))}


def level_camera_axes(azimuth_rad, elevation_rad):
    """Image right and image up (earth coordinates) of a camera without roll looking at the given azimuth and elevation."""
    forward = directions_from_zenith_azimuth(np.pi / 2 - elevation_rad, azimuth_rad)
    level_right = np.array([np.cos(azimuth_rad), -np.sin(azimuth_rad), 0.0])
    return level_right, np.cross(level_right, forward)


def camera_orientation_from_device(alpha_deg, beta_deg, gamma_deg, screen_angle_deg=0.0, **kwargs):
    """Orientation of the phone's back camera (viewing azimuth, elevation, roll in degrees) from absolute device orientation angles.

    `screen_angle_deg` is the screen's rotation (`screen.orientation.angle`: 0 portrait, 90 landscape with the top of
    the phone to the left, 270 to the right); the image's right and up follow the screen. The back camera looks along
    the device's -z axis.
    """
    rotation = rotation_from_device_angles(alpha_deg, beta_deg, gamma_deg)
    screen_angle = np.radians(screen_angle_deg)
    right_in_device = np.array([np.cos(screen_angle), -np.sin(screen_angle), 0.0])
    up_in_device = np.array([np.sin(screen_angle), np.cos(screen_angle), 0.0])
    return camera_orientation_from_axes(rotation @ right_in_device, rotation @ up_in_device, rotation @ np.array([0.0, 0.0, -1.0]))


class CameraView:
    """Pinhole camera at the panel position: viewing direction, roll and field of view, for an image of the given size in pixels.

    `fov_deg` is the field of view across the image's longer side, a property of the phone's camera independent of
    portrait or landscape. Sky patch edges are great circle arcs, which a pinhole camera maps to straight lines, so the
    projected corners of a patch give its exact outline in the photo.
    """

    def __init__(self, azimuth_deg=180.0, elevation_deg=30.0, roll_deg=0.0, fov_deg=65.0, width=1000, height=750, **kwargs):
        if not 0.0 < fov_deg < 180.0:
            raise ValueError(f"Camera field of view must be between 0 and 180 degrees, got {fov_deg}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Image size must be positive, got {width} x {height} pixels")
        self.azimuth_deg, self.elevation_deg, self.roll_deg, self.fov_deg = float(azimuth_deg), float(elevation_deg), float(roll_deg), float(fov_deg)
        self.width, self.height = float(width), float(height)
        azimuth_rad, elevation_rad, roll_rad = np.radians([azimuth_deg, elevation_deg, roll_deg])
        self.forward = directions_from_zenith_azimuth(np.pi / 2 - elevation_rad, azimuth_rad)
        level_right, level_up = level_camera_axes(azimuth_rad, elevation_rad)
        self.right = np.cos(roll_rad) * level_right + np.sin(roll_rad) * level_up
        self.up = np.cross(self.right, self.forward)
        self.focal_length_px = max(self.width, self.height) / 2.0 / np.tan(np.radians(fov_deg) / 2.0)

    def settings(self):
        """The view's parameters as a plain dict (e.g. for the metadata)."""
        return {"azimuth_deg": self.azimuth_deg, "elevation_deg": self.elevation_deg, "roll_deg": self.roll_deg, "fov_deg": self.fov_deg, "width": self.width, "height": self.height}

    def project(self, directions):
        """Pixel coordinates (x right, y down from the top left corner) of directions of shape (n, 3), and whether each lies in front of the camera."""
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        depth = directions @ self.forward
        in_front = depth > 1e-9 * np.linalg.norm(directions, axis=1)
        safe_depth = np.where(in_front, depth, 1.0)
        pixel_x = self.width / 2.0 + self.focal_length_px * (directions @ self.right) / safe_depth
        pixel_y = self.height / 2.0 - self.focal_length_px * (directions @ self.up) / safe_depth
        return np.stack([pixel_x, pixel_y], axis=1), in_front

    def directions(self, pixels):
        """Unit directions (earth coordinates) seen at pixel coordinates of shape (n, 2); inverse of `project`."""
        pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
        across = (pixels[:, 0] - self.width / 2.0) / self.focal_length_px
        upwards = (self.height / 2.0 - pixels[:, 1]) / self.focal_length_px
        directions = self.forward + across[:, None] * self.right + upwards[:, None] * self.up
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)
