"""Point cloud from an AR video recording (#46, #48): feature matching and triangulation with the known camera poses.

ARCore delivers each keyframe's camera pose in metres (`core.ar_recording`), so no pose estimation is needed: features
are detected per frame (ORB by default), matched between frame pairs with enough baseline and overlapping views
(only candidates near the epipolar line of the known poses), joined into tracks over several frames and triangulated.
Points without measurable parallax (rays nearly parallel, e.g. clouds or distant terrain) are discarded. The result is
turned to north with the recording's heading offset: x east, y north, z up in metres, origin at the reference
position (the recording's first pose). Per point it holds the number of observing frames, the triangulation angle and
the estimated distance error, so the obstruction step (#49) can filter.

OpenCV is imported on first use: in the browser it is a 10.7 MB download, loaded only for this step.

Camera conventions: WebXR cameras look along their -z axis with y up (`core.ar_recording`); OpenCV cameras look along
+z with y down. Pixel coordinates have x to the right and y down from the image's top left corner.
"""

import datetime
import io
import logging
import time

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pv_yield_estimator import __version__
from pv_yield_estimator.core.ar_recording import ar_to_earth, camera_path, frame_rotations, recording_heading
from pv_yield_estimator.file_format import FORMAT_VERSION_KEY, check_format_version

log = logging.getLogger(__name__)

POINTS_FORMAT_VERSION = 1
POINTS_KIND = "reconstructed_points"
POINT_COLUMNS = ["east_m", "north_m", "up_m", "n_frames", "triangulation_angle_deg", "distance_m", "distance_error_m", "reprojection_error_px"]
### WebXR camera axes (x right, y up, z backward) to OpenCV camera axes (x right, y down, z forward).
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def camera_matrix(projection_matrix, width, height):
    """Pinhole camera matrix K (3 × 3, pixels) of an image of `width` × `height` pixels from a WebXR projection matrix (16 entries, column-major)."""
    matrix = np.asarray(projection_matrix, dtype=float)
    ### Normalized device x = P00 x / d - P02 and y = P11 y / d - P12 at depth d; pixels from x right and y down.
    return np.array([[matrix[0] * width / 2, 0.0, width / 2 * (1 - matrix[8])], [0.0, matrix[5] * height / 2, height / 2 * (1 + matrix[9])], [0.0, 0.0, 1.0]])


class FramePoses:
    """Camera poses of a recording's frames for OpenCV: world (AR coordinates) to camera rotations, translations, camera centres, viewing directions and camera matrices for the processed image sizes."""

    def __init__(self, recording, image_sizes):
        frames = recording["frames"]
        rotations = frame_rotations(recording)                                         ### camera (WebXR axes) to AR
        self.world_to_camera = GL_TO_CV @ np.transpose(rotations, (0, 2, 1))          ### AR to camera (OpenCV axes)
        self.centres = np.array([frame["position_m"] for frame in frames], dtype=float).reshape(-1, 3)
        self.translations = -np.einsum("nij,nj->ni", self.world_to_camera, self.centres)
        self.forward = -rotations[:, :, 2]
        self.camera_matrices = np.array([camera_matrix(frame["projection_matrix"], width, height) for frame, (width, height) in zip(frames, image_sizes)]).reshape(-1, 3, 3)

    def fundamental_matrix(self, first, second):
        """Fundamental matrix F of a frame pair from the known poses: x2ᵀ F x1 = 0 for pixels x1 in the first and x2 in the second frame."""
        rotation = self.world_to_camera[second] @ self.world_to_camera[first].T
        translation = self.translations[second] - rotation @ self.translations[first]
        cross = np.array([[0.0, -translation[2], translation[1]], [translation[2], 0.0, -translation[0]], [-translation[1], translation[0], 0.0]])
        return np.linalg.inv(self.camera_matrices[second]).T @ cross @ rotation @ np.linalg.inv(self.camera_matrices[first])

    def rays(self, frame_indices, pixels):
        """Unit viewing directions (AR coordinates) through pixels (n, 2) of the given frames (n)."""
        homogeneous = np.concatenate([pixels, np.ones((len(pixels), 1))], axis=1)
        in_camera = np.einsum("nij,nj->ni", np.linalg.inv(self.camera_matrices)[frame_indices], homogeneous)
        directions = np.einsum("nji,nj->ni", self.world_to_camera[frame_indices], in_camera)
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)

    def project(self, frame_indices, points):
        """Pixels (n, 2) and depths (n) of points (n, 3, AR coordinates) in the given frames (n)."""
        in_camera = np.einsum("nij,nj->ni", self.world_to_camera[frame_indices], points) + self.translations[frame_indices]
        depth = in_camera[:, 2]
        homogeneous = np.einsum("nij,nj->ni", self.camera_matrices[frame_indices], in_camera)
        return homogeneous[:, :2] / np.where(np.abs(depth) > 1e-12, depth, 1e-12)[:, None], depth


def decode_frame(image_bytes, max_image_px=960):
    """Grayscale image (uint8) from JPEG bytes, downscaled so that its longer side is at most `max_image_px` (None: full size)."""
    import cv2
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Frame image could not be decoded ({len(image_bytes)} bytes, starting with {bytes(image_bytes[:8])!r})")
    scale = 1.0 if max_image_px is None else min(1.0, max_image_px / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return image


def feature_detector(feature="orb", max_features=2000):
    """OpenCV feature detector and descriptor (`orb` or `sift`) and the norm to match its descriptors with."""
    import cv2
    if feature == "orb":
        return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING
    if feature == "sift":
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    raise ValueError(f"Unknown feature type {feature!r}; known: orb, sift")


def select_frame_pairs(poses, usable, fov_deg, min_baseline_m=0.2, max_view_angle_deg=None, max_pairs_per_frame=5):
    """Frame pairs (i < j) to match: both usable, at least `min_baseline_m` apart and viewing directions at most `max_view_angle_deg` apart (default: half the field of view).

    Of the candidates of each frame, at most `max_pairs_per_frame` partners are kept, spread evenly over the range of
    baselines (short ones match more easily, long ones triangulate more precisely). Returns an array (n, 2).
    """
    max_view_angle_deg = fov_deg / 2 if max_view_angle_deg is None else max_view_angle_deg
    baselines = np.linalg.norm(poses.centres[:, None] - poses.centres[None], axis=2)
    view_angles = np.degrees(np.arccos(np.clip(poses.forward @ poses.forward.T, -1.0, 1.0)))
    ### Without any baseline the epipolar geometry is undefined.
    candidates = (baselines >= max(min_baseline_m, 1e-6)) & (view_angles <= max_view_angle_deg) & usable[:, None] & usable[None] & np.triu(np.ones_like(baselines, dtype=bool), k=1)
    pairs = []
    for first in range(len(baselines)):
        partners = np.flatnonzero(candidates[first])
        if len(partners) > max_pairs_per_frame:
            partners = partners[np.argsort(baselines[first, partners])][np.unique(np.round(np.linspace(0, len(partners) - 1, max_pairs_per_frame)).astype(int))]
        pairs += [(first, second) for second in partners]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def match_pair(descriptors_first, descriptors_second, norm, ratio=0.8, mask=None):
    """Descriptor matches (n, 2: index in the first, index in the second frame) passing the ratio test, each feature of the second frame used at most once (the best match).

    `mask` (uint8, features of the first × features of the second frame) restricts the candidates, e.g. to the
    epipolar band; OpenCV then skips the other descriptor distances.
    """
    import cv2
    if descriptors_first is None or descriptors_second is None or len(descriptors_first) < 2 or len(descriptors_second) < 2:
        return np.zeros((0, 2), dtype=int)
    knn = cv2.BFMatcher(norm).knnMatch(descriptors_first, descriptors_second, k=2, mask=mask if mask is not None else np.ones((len(descriptors_first), len(descriptors_second)), dtype=np.uint8))
    good = [(best.queryIdx, best.trainIdx, best.distance) for best, second in (pair for pair in knn if len(pair) == 2) if best.distance < ratio * second.distance]
    if not good:
        return np.zeros((0, 2), dtype=int)
    good = np.array(good)
    order = np.lexsort((good[:, 2], good[:, 1]))
    good = good[order]
    first_of_each = np.concatenate([[True], good[1:, 1] != good[:-1, 1]])
    return good[first_of_each, :2].astype(int)


def epipolar_band(fundamental, pixels_first, pixels_second, max_epipolar_px=2.0):
    """Mask (uint8, features of the first × features of the second frame) of the feature pairs whose second pixel lies within `max_epipolar_px` of the first one's epipolar line."""
    lines = np.concatenate([pixels_first, np.ones((len(pixels_first), 1))], axis=1) @ fundamental.T
    lines /= np.maximum(np.hypot(lines[:, 0], lines[:, 1]), 1e-12)[:, None]
    ### Single precision and in place: this dense product is a large share of the matching time.
    distances = lines.astype(np.float32) @ np.concatenate([pixels_second, np.ones((len(pixels_second), 1))], axis=1).T.astype(np.float32)
    np.abs(distances, out=distances)
    return (distances <= max_epipolar_px).view(np.uint8)


def epipolar_distances(fundamental, pixels_first, pixels_second):
    """Larger of the two distances (pixels) of matched pixels from each other's epipolar line."""
    first = np.concatenate([pixels_first, np.ones((len(pixels_first), 1))], axis=1)
    second = np.concatenate([pixels_second, np.ones((len(pixels_second), 1))], axis=1)
    lines_second, lines_first = first @ fundamental.T, second @ fundamental
    residual = np.abs(np.sum(second * lines_second, axis=1))
    return np.maximum(residual / np.hypot(lines_second[:, 0], lines_second[:, 1]), residual / np.hypot(lines_first[:, 0], lines_first[:, 1]))


def triangulate_tracks(track_ids, centres, rays, n_tracks):
    """Least-squares intersection of the rays of each track (the point closest to all its rays), shape (n_tracks, 3)."""
    projectors = np.eye(3) - rays[:, :, None] * rays[:, None, :]
    normal_matrices = np.zeros((n_tracks, 3, 3))
    right_sides = np.zeros((n_tracks, 3))
    np.add.at(normal_matrices, track_ids, projectors)
    np.add.at(right_sides, track_ids, np.einsum("nij,nj->ni", projectors, centres))
    ### Rays all parallel make the matrix singular; the tiny regularization keeps the solve finite, the angle check drops these.
    return np.linalg.solve(normal_matrices + 1e-9 * np.eye(3), right_sides[:, :, None])[:, :, 0]


def track_angles(track_ids, directions, n_tracks):
    """Largest angle (degrees) between the directions from a track's point to its observing cameras, from the direction farthest from their mean (exact for cameras along a line)."""
    mean = np.zeros((n_tracks, 3))
    np.add.at(mean, track_ids, directions)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    to_mean = np.einsum("ni,ni->n", directions, mean[track_ids])
    order = np.lexsort((to_mean, track_ids))
    starts = np.flatnonzero(np.concatenate([[True], track_ids[order][1:] != track_ids[order][:-1]]))
    farthest = np.zeros((n_tracks, 3))
    farthest[track_ids[order][starts]] = directions[order][starts]
    cosines = np.einsum("ni,ni->n", directions, farthest[track_ids])
    minimum = np.full(n_tracks, 1.0)
    np.minimum.at(minimum, track_ids, cosines)
    return np.degrees(np.arccos(np.clip(minimum, -1.0, 1.0)))


def reconstruct_points(recording, frame_images, heading_offset_deg=None, feature="orb", max_features=2000, max_image_px=960, min_baseline_m=0.2, max_view_angle_deg=None, max_pairs_per_frame=5,
                       ratio=0.8, max_epipolar_px=2.0, max_reprojection_px=2.0, min_triangulation_angle_deg=2.0, min_frames=2, min_distance_m=0.3, pixel_error_px=1.0, **kwargs):
    """Point cloud from an AR recording (`recording.json` as dict) and its frames' JPEG bytes (same order); returns `ReconstructedPoints`.

    Settings (config section `sky_obstruction.ar_video`):
    - `heading_offset_deg`: turns the AR coordinates to north; default as in the inspection view (set by hand, else
      from the compass).
    - `feature` (`orb` or `sift`), `max_features` per frame, `max_image_px`: frames are downscaled to this longer side
      (None: full size).
    - Frame pairs (`select_frame_pairs`): `min_baseline_m`, `max_view_angle_deg` (default half the field of view),
      `max_pairs_per_frame`.
    - Matching (guided by the known poses): candidates within `max_epipolar_px` of the epipolar line, Lowe's `ratio`
      test among them.
    - Points: triangulated per track (matches joined over all frames), observations off by more than
      `max_reprojection_px` dropped once, then at least `min_frames` observing frames and a triangulation angle of at
      least `min_triangulation_angle_deg`, at least `min_distance_m` from the observing cameras (on average; rays from
      nearly the same position, e.g. when turning on the spot, meet at the camera). `pixel_error_px` is the assumed feature position error for the estimated
      distance error.
    Frames with lost tracking (emulated position) are not used.
    """
    timings = {}
    start = time.perf_counter()
    frames = recording["frames"]
    if len(frame_images) != len(frames):
        raise ValueError(f"The AR recording has {len(frames)} frames but {len(frame_images)} images were given")
    heading = recording_heading(recording, heading_offset_deg)
    usable = np.array([not frame.get("emulated_position") for frame in frames], dtype=bool)
    images = [decode_frame(image, max_image_px) if use else None for image, use in zip(frame_images, usable)]
    image_sizes = [(image.shape[1], image.shape[0]) if image is not None else (frame["width"], frame["height"]) for image, frame in zip(images, frames)]
    poses = FramePoses(recording, image_sizes)
    timings["decode_s"] = time.perf_counter() - start

    start = time.perf_counter()
    detector, norm = feature_detector(feature, max_features)
    keypoints, descriptors = [], []
    for image in images:
        found, described = detector.detectAndCompute(image, None) if image is not None else ((), None)
        keypoints.append(np.array([point.pt for point in found], dtype=float).reshape(-1, 2))
        descriptors.append(described)
    timings["features_s"] = time.perf_counter() - start

    start = time.perf_counter()
    fov_deg = float(np.median([np.degrees(2 * np.arctan(max(size) / 2 / max(matrix[0, 0], matrix[1, 1]))) for size, matrix in zip(image_sizes, poses.camera_matrices)])) if frames else 60.0
    pairs = select_frame_pairs(poses, usable, fov_deg, min_baseline_m, max_view_angle_deg, max_pairs_per_frame)
    offsets = np.concatenate([[0], np.cumsum([len(points) for points in keypoints])])
    edges, n_matches = [], 0
    for first, second in pairs:
        fundamental = poses.fundamental_matrix(first, second)
        matches = match_pair(descriptors[first], descriptors[second], norm, ratio, epipolar_band(fundamental, keypoints[first], keypoints[second], max_epipolar_px))
        n_matches += len(matches)
        if len(matches):
            distances = epipolar_distances(fundamental, keypoints[first][matches[:, 0]], keypoints[second][matches[:, 1]])
            matches = matches[distances <= max_epipolar_px]
            edges.append(np.stack([matches[:, 0] + offsets[first], matches[:, 1] + offsets[second]], axis=1))
    edges = np.concatenate(edges) if edges else np.zeros((0, 2), dtype=int)
    timings["matching_s"] = time.perf_counter() - start

    start = time.perf_counter()
    points, observation_count = triangulate_matches(edges, offsets, keypoints, poses, max_reprojection_px, min_triangulation_angle_deg, min_frames, min_distance_m, pixel_error_px)
    to_earth = ar_to_earth(heading["heading_offset_deg"])
    points[["east_m", "north_m", "up_m"]] = points[["east_m", "north_m", "up_m"]].to_numpy() @ to_earth.T
    timings["triangulation_s"] = time.perf_counter() - start

    statistics = {"n_frames": len(frames), "n_frames_used": int(usable.sum()), "n_features": int(offsets[-1]), "n_pairs": len(pairs), "n_matches": n_matches, "n_verified_matches": len(edges),
                  "n_observations": observation_count, "n_points": len(points), **{key: round(value, 3) for key, value in timings.items()}}
    settings = {"feature": feature, "max_features": max_features, "max_image_px": max_image_px, "min_baseline_m": min_baseline_m, "max_view_angle_deg": max_view_angle_deg, "max_pairs_per_frame": max_pairs_per_frame,
                "ratio": ratio, "max_epipolar_px": max_epipolar_px, "max_reprojection_px": max_reprojection_px, "min_triangulation_angle_deg": min_triangulation_angle_deg, "min_frames": min_frames, "min_distance_m": min_distance_m,
                "pixel_error_px": pixel_error_px}
    metadata = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "program_version": __version__, "recording_created": recording.get("created"), **heading,
                "settings": settings, "statistics": statistics}
    log.info(f"Point cloud from the AR recording: {len(points)} points from {statistics['n_frames_used']} frames, {len(pairs)} frame pairs, {len(edges)} of {n_matches} matches consistent with the poses; "
             f"{sum(timings.values()):.1f} s (" + ", ".join(f"{key[:-2]} {value:.1f} s" for key, value in timings.items()) + ")")
    return ReconstructedPoints(points, metadata, camera_path(recording, heading["heading_offset_deg"]))


def triangulate_matches(edges, offsets, keypoints, poses, max_reprojection_px=2.0, min_triangulation_angle_deg=2.0, min_frames=2, min_distance_m=0.3, pixel_error_px=1.0):
    """Join the verified matches (pairs of global feature numbers) into tracks and triangulate them; returns the points table (AR coordinates) and the number of observations used."""
    if len(edges) == 0:
        return pd.DataFrame(columns=POINT_COLUMNS, dtype=float), 0
    n_features = offsets[-1]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_features, n_features))
    _, labels = connected_components(graph, directed=False)
    features = np.unique(edges)
    frame_of_feature = np.searchsorted(offsets, features, side="right") - 1
    pixels = np.concatenate(keypoints)[features]
    _, track_ids = np.unique(labels[features], return_inverse=True)
    n_tracks = track_ids.max() + 1
    rays = poses.rays(frame_of_feature, pixels)
    centres = poses.centres[frame_of_feature]
    keep = np.ones(len(features), dtype=bool)
    ### Triangulate, drop the observations off by more than the limit (wrong matches joined into the track), twice, then triangulate the rest.
    for final in (False, False, True):
        points = triangulate_tracks(track_ids[keep], centres[keep], rays[keep], n_tracks)
        projected, depth = poses.project(frame_of_feature, points[track_ids])
        errors = np.linalg.norm(projected - pixels, axis=1)
        if not final:
            keep &= (errors <= max_reprojection_px) & (depth > 0)
    ### Distinct observing frames per track; a track seen twice in a frame counts that frame once.
    track_frames = np.unique(np.stack([track_ids[keep], frame_of_feature[keep]], axis=1), axis=0)
    n_frames = np.bincount(track_frames[:, 0], minlength=n_tracks)
    to_cameras = centres[keep] - points[track_ids[keep]]
    distances = np.linalg.norm(to_cameras, axis=1)
    angles = track_angles(track_ids[keep], to_cameras / np.maximum(distances, 1e-12)[:, None], n_tracks)
    observations = np.maximum(np.bincount(track_ids[keep], minlength=n_tracks), 1)
    mean_distance = np.bincount(track_ids[keep], weights=distances, minlength=n_tracks) / observations
    max_error = np.zeros(n_tracks)
    np.maximum.at(max_error, track_ids[keep], errors[keep])
    focal_px = np.median(poses.camera_matrices[:, 0, 0])
    distance_error = mean_distance * (pixel_error_px / focal_px) / np.maximum(np.sin(np.radians(angles)), 1e-12)
    valid = (n_frames >= min_frames) & (angles >= min_triangulation_angle_deg) & (mean_distance >= min_distance_m)
    table = pd.DataFrame({"east_m": points[valid, 0], "north_m": points[valid, 1], "up_m": points[valid, 2], "n_frames": n_frames[valid], "triangulation_angle_deg": angles[valid],
                          "distance_m": mean_distance[valid], "distance_error_m": distance_error[valid], "reprojection_error_px": max_error[valid]})
    log.debug(f"Triangulation: {n_tracks} tracks, {np.count_nonzero(keep)} of {len(keep)} observations within {max_reprojection_px} px, {np.count_nonzero(valid)} points kept")
    return table, int(np.count_nonzero(keep & valid[track_ids]))


class ReconstructedPoints:
    """Point cloud reconstructed from an AR recording: a table with one row per point (`POINT_COLUMNS`; x east, y north, z up in metres from the reference position), the metadata (settings, statistics, heading offset) and the camera path (earth coordinates).

    Stored as CSV (`reconstructed_points.csv` in the project zip) with the metadata as YAML in comment lines at the top;
    the camera path is not stored, it comes from the recording.
    """

    def __init__(self, table, metadata=None, path=None):
        self.table = table.reset_index(drop=True)
        self.metadata = dict(metadata or {})
        self.path = np.zeros((0, 3)) if path is None else np.asarray(path, dtype=float)

    @property
    def points(self):
        """Point positions (n, 3): east, north, up in metres from the reference position."""
        return self.table[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)

    def to_csv_text(self):
        """CSV text: comment lines with format version, kind and metadata (YAML), then the table."""
        header = yaml.safe_dump({FORMAT_VERSION_KEY: POINTS_FORMAT_VERSION, "kind": POINTS_KIND, **self.metadata}, sort_keys=False, default_flow_style=False, allow_unicode=True)
        comments = "".join(f"# {line}\n" for line in header.splitlines())
        return comments + self.table.to_csv(index=False, float_format="%.4f", lineterminator="\n")

    @classmethod
    def from_csv_text(cls, text, source="<reconstructed points>"):
        """Read the CSV text written by `to_csv_text`, checking format version and kind."""
        header = yaml.safe_load("\n".join(line[2:] for line in text.splitlines() if line.startswith("#"))) or {}
        check_format_version(header, POINTS_FORMAT_VERSION, "Reconstructed points", source)
        if header.get("kind") != POINTS_KIND:
            raise ValueError(f"{source} has kind {header.get('kind')!r}, expected {POINTS_KIND!r}")
        table = pd.read_csv(io.StringIO(text), comment="#")
        missing = [column for column in POINT_COLUMNS if column not in table.columns]
        if missing:
            raise ValueError(f"{source} lacks the columns {', '.join(missing)}")
        return cls(table, {key: value for key, value in header.items() if key not in (FORMAT_VERSION_KEY, "kind")})
