"""AR video recordings (#46, #47): keyframes with camera poses from ARCore (WebXR) and compass readings.

A recording is made in the browser on an Android phone (`web/ar_recording.js`): an `immersive-ar` WebXR session with
camera access. For each kept frame (keyframe) it stores the camera image (JPEG, a separate file), the camera pose in
the session's `local` reference space and the projection matrix, plus the latest compass reading. This module checks
the stored format and converts the poses into the photo method's camera views, so the frames can be shown on the sky
map like photos. The point cloud from a recording follows in #48.

AR coordinates (WebXR `local` reference space): metres, origin at the phone's position when the session started (the
reference position), y up, -z the horizontal direction the camera faced at the start, x to its right. The camera of a
view looks along its own -z axis, with x to the right of the image and y up. Earth coordinates as in `sky`: x east,
y north, z up. The two differ by a turn about the vertical, given by the heading offset: the compass azimuth of the
AR -z axis (the direction the camera faced at the start).
"""

import logging

import numpy as np

from pv_yield_estimator.core.photo import CameraView, camera_orientation_from_axes, rotation_from_device_angles
from pv_yield_estimator.core.weather import distance_km
from pv_yield_estimator.file_format import FORMAT_VERSION_KEY, check_format_version

log = logging.getLogger(__name__)

RECORDING_FORMAT_VERSION = 1
RECORDING_KIND = "ar_recording"
FRAME_KEYS = ("file", "time_s", "position_m", "orientation_xyzw", "width", "height", "projection_matrix")
### AR coordinates to earth coordinates for a heading offset of 0 (AR -z points north): x -> east, y -> up, z -> south.
AR_TO_EARTH_NORTH = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def frame_file_name(number):
    """File name of a frame's JPEG in the project zip (numbers start at 1), e.g. recording/frame_1.jpg."""
    return f"recording/frame_{number}.jpg"


def check_recording(data, source="<recording>"):
    """Check a recording's dict (content of `recording.json`): format version, kind and the entries of each frame; returns it."""
    check_format_version(data, RECORDING_FORMAT_VERSION, "AR recording", source)
    if data.get("kind") != RECORDING_KIND:
        raise ValueError(f"{source} has kind {data.get('kind')!r}, expected {RECORDING_KIND!r}")
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{source}: 'frames' must be a list, got {type(frames).__name__}")
    for number, frame in enumerate(frames, start=1):
        missing = [key for key in FRAME_KEYS if key not in frame]
        if missing:
            raise ValueError(f"{source}: frame {number} ({frame.get('file', '?')}) lacks {', '.join(missing)}")
        if len(frame["position_m"]) != 3 or len(frame["orientation_xyzw"]) != 4 or len(frame["projection_matrix"]) != 16:
            raise ValueError(f"{source}: frame {number} ({frame['file']}) needs 3 position entries, 4 quaternion entries and 16 projection matrix entries")
    return data


def new_recording(frames=(), **details):
    """A recording dict in the current format with the given frames and further top-level entries (e.g. gps, reference_note)."""
    return {FORMAT_VERSION_KEY: RECORDING_FORMAT_VERSION, "kind": RECORDING_KIND, **details, "frames": list(frames)}


def rotation_from_quaternion(quaternions_xyzw):
    """Rotation matrices (n, 3, 3) from unit quaternions (n, 4) in WebXR's order x, y, z, w."""
    quaternions = np.atleast_2d(np.asarray(quaternions_xyzw, dtype=float))
    x, y, z, w = (quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)


def ar_to_earth(heading_offset_deg):
    """Rotation from AR to earth coordinates: the AR -z axis turned to the compass azimuth `heading_offset_deg`."""
    angle = np.radians(-heading_offset_deg)              ### counterclockwise seen from above
    turn = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    return turn @ AR_TO_EARTH_NORTH


def fov_from_projection_matrix(projection_matrix, width, height):
    """Field of view (degrees) across the image's longer side from a WebXR projection matrix (16 entries, column-major)."""
    matrix = np.asarray(projection_matrix, dtype=float)
    scale = matrix[0] if width >= height else matrix[5]                    ### 1 / tan(half the field of view)
    return float(np.degrees(2.0 * np.arctan(1.0 / scale)))


def frame_rotations(recording):
    """Camera rotations (camera to AR coordinates) of all frames, shape (n, 3, 3); the columns are the image right, image up and the camera's backward direction."""
    return rotation_from_quaternion([frame["orientation_xyzw"] for frame in recording["frames"]]).reshape(-1, 3, 3)


def compass_camera_rotation(compass, screen_angle_deg=0.0):
    """Camera rotation (camera to earth coordinates) from a compass reading (absolute device orientation angles) and the screen's rotation, columns as in `frame_rotations`."""
    angle = np.radians(screen_angle_deg)
    camera_in_device = np.array([[np.cos(angle), np.sin(angle), 0.0], [-np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    return rotation_from_device_angles(compass["alpha_deg"], compass["beta_deg"], compass["gamma_deg"]) @ camera_in_device


def heading_offset_from_rotations(earth_rotation, ar_rotation):
    """Heading offset (degrees) that turns a camera rotation in AR coordinates into one in earth coordinates: from the turn about the vertical between the two, robust also with the camera looking up or down."""
    difference = earth_rotation @ (AR_TO_EARTH_NORTH @ ar_rotation).T
    yaw = np.arctan2(difference[1, 0] - difference[0, 1], difference[0, 0] + difference[1, 1])
    return float(np.degrees(-yaw) % 360.0)


def robust_mean_angle(angles_deg):
    """Median of angles (degrees) around their circular mean, and the median absolute deviation from it."""
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    mean = np.arctan2(np.sin(angles).sum(), np.cos(angles).sum())
    deviations = np.degrees(np.angle(np.exp(1j * (angles - mean))))
    median = np.median(deviations)
    return float((np.degrees(mean) + median) % 360.0), float(np.median(np.abs(deviations - median)))


def heading_offset_from_compass(recording, max_age_s=0.5, max_gap_s=3.0):
    """Estimate the heading offset from the compass readings: the robust mean of compass heading minus AR heading over the frames.

    Each frame's latest compass reading counts if it is at most `max_age_s` older than the frame. Without any (if the
    compass does not report during the AR session), the readings just before and after the session are paired with
    the first and the last frame, if at most `max_gap_s` apart. Returns a dict with the offset (None without
    readings), the number of readings used, their spread (median absolute deviation, degrees) and the source.
    """
    frames = recording["frames"]
    if not frames:
        return {"heading_offset_deg": None, "n_readings": 0, "spread_deg": None, "source": "none"}
    rotations = frame_rotations(recording)
    pairs = [(frame["compass"], frame.get("screen_angle_deg", 0.0), rotation) for frame, rotation in zip(frames, rotations)
             if frame.get("compass") and frame["time_s"] - frame["compass"]["time_s"] <= max_age_s]
    source = "compass"
    if not pairs:
        source = "compass_before_after"
        for compass, frame, rotation in ((recording.get("compass_before"), frames[0], rotations[0]), (recording.get("compass_after"), frames[-1], rotations[-1])):
            if compass and abs(frame["time_s"] - compass["time_s"]) <= max_gap_s:
                pairs.append((compass, compass.get("screen_angle_deg", frame.get("screen_angle_deg", 0.0)), rotation))
    if not pairs:
        return {"heading_offset_deg": None, "n_readings": 0, "spread_deg": None, "source": "none"}
    offsets = [heading_offset_from_rotations(compass_camera_rotation(compass, screen_angle), rotation) for compass, screen_angle, rotation in pairs]
    offset, spread = robust_mean_angle(offsets)
    return {"heading_offset_deg": offset, "n_readings": len(offsets), "spread_deg": spread, "source": source}


def frame_camera_views(recording, heading_offset_deg):
    """The frames as camera views of the photo method (`CameraView`): viewing azimuth, elevation, roll, field of view and image size, with the AR orientation turned to north."""
    rotations = ar_to_earth(heading_offset_deg) @ frame_rotations(recording)
    views = []
    for frame, rotation in zip(recording["frames"], rotations):
        orientation = camera_orientation_from_axes(rotation[:, 0], rotation[:, 1], -rotation[:, 2])
        views.append(CameraView(**orientation, fov_deg=fov_from_projection_matrix(frame["projection_matrix"], frame["width"], frame["height"]), width=frame["width"], height=frame["height"]))
    return views


def camera_path(recording, heading_offset_deg):
    """Camera positions of the frames in metres from the reference position (the AR origin), in earth coordinates (east, north, up), shape (n, 3)."""
    positions = np.array([frame["position_m"] for frame in recording["frames"]], dtype=float).reshape(-1, 3)
    return positions @ ar_to_earth(heading_offset_deg).T


def hemisphere_directions(n_directions=20_000):
    """Directions spread evenly over the upper hemisphere (equal solid angle each, Fibonacci spiral), shape (n, 3)."""
    index = np.arange(n_directions)
    z = (index + 0.5) / n_directions
    azimuth = index * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.sqrt(1.0 - z * z)
    return np.stack([radius * np.sin(azimuth), radius * np.cos(azimuth), z], axis=1)


def covered_fraction(views, n_directions=20_000):
    """Share of the sky hemisphere's solid angle seen in at least one of the camera views."""
    directions = hemisphere_directions(n_directions)
    covered = np.zeros(len(directions), dtype=bool)
    for view in views:
        pixels, in_front = view.project(directions)
        covered |= in_front & (pixels[:, 0] >= 0) & (pixels[:, 0] <= view.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= view.height)
    return float(covered.mean())


def projection_diagnostics(frame):
    """How the projection matrix fits the camera image: field of view across and along, the principal point's offset from the image centre (pixels) and the aspect ratios of image and projection."""
    matrix = np.asarray(frame["projection_matrix"], dtype=float)
    width, height = frame["width"], frame["height"]
    return {"fov_horizontal_deg": float(np.degrees(2 * np.arctan(1 / matrix[0]))), "fov_vertical_deg": float(np.degrees(2 * np.arctan(1 / matrix[5]))),
            "principal_point_offset_px": [float(matrix[8] * width / 2), float(matrix[9] * height / 2)],
            "image_aspect": float(width / height), "projection_aspect": float(matrix[5] / matrix[0])}


def recording_summary(recording, heading_offset_deg=None, site_latitude=None, site_longitude=None):
    """Key figures of a recording for the inspection view, its frames as camera views and its camera path.

    The heading offset is the given one (set by hand), else the recording's stored one if set by hand, else the
    estimate from the compass (0 without readings). Returns a dict: numbers of frames and of frames with lost
    tracking, duration, the path's extent (baseline: diagonal of its bounding box), length and start-to-end distance,
    the heading offset with its source, the covered share of the sky, the GPS position's distance from the site (if
    both are known), and per frame the camera view (`views`) and position (`path`, east/north/up in metres).
    """
    frames = recording["frames"]
    estimate = heading_offset_from_compass(recording)
    if heading_offset_deg is None and recording.get("heading_source") == "manual":
        heading_offset_deg = recording.get("heading_offset_deg")
    if heading_offset_deg is not None:
        heading = {"heading_offset_deg": float(heading_offset_deg) % 360.0, "heading_source": "manual"}
    else:
        heading = {"heading_offset_deg": estimate["heading_offset_deg"] or 0.0, "heading_source": estimate["source"]}
    views = frame_camera_views(recording, heading["heading_offset_deg"])
    path = camera_path(recording, heading["heading_offset_deg"])
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.zeros(0)
    gps = recording.get("gps") or {}
    gps_distance = distance_km(site_latitude, site_longitude, gps["latitude"], gps["longitude"]) if site_latitude is not None and site_longitude is not None and gps.get("latitude") is not None else None
    summary = {"n_frames": len(frames), "n_tracking_lost": sum(bool(frame.get("emulated_position")) for frame in frames),
               "duration_s": float(frames[-1]["time_s"] - frames[0]["time_s"]) if frames else 0.0,
               "baseline_m": float(np.linalg.norm(path.max(axis=0) - path.min(axis=0))) if len(path) else 0.0,
               "path_length_m": float(steps.sum()), "start_to_end_m": float(np.linalg.norm(path[-1] - path[0])) if len(path) else 0.0,
               **heading, "compass_estimate": estimate, "covered_fraction": covered_fraction(views), "gps_distance_km": gps_distance,
               "projection": projection_diagnostics(frames[0]) if frames else None,
               "views": [view.settings() for view in views], "path": np.round(path, 4).tolist()}
    log.info(f"AR recording: {summary['n_frames']} frames over {summary['duration_s']:.1f} s, baseline {summary['baseline_m']:.2f} m, heading offset {heading['heading_offset_deg']:.1f}° ({heading['heading_source']})")
    return summary
