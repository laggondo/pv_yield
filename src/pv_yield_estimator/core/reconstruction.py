"""Point cloud from an AR video recording (#46, #48): feature matching and triangulation with the known camera poses.

ARCore delivers each keyframe's camera pose in metres (`core.ar_recording`), so no pose estimation is needed: features
are detected per frame (ORB by default; blurred frames are skipped), matched between frame pairs with enough baseline
and overlapping views (candidates near the epipolar line of the recorded poses, then checked against the images
alone), joined into tracks over several frames, the poses refined from the images (bundle adjustment: ARCore's
orientations can be off by about half a degree, which shifts points at a few metres by up to a metre with short
baselines) and the tracks triangulated.
Points without measurable parallax (rays nearly parallel, e.g. clouds or distant terrain) are discarded. The result is
turned to north with the recording's heading offset: x east, y north, z up in metres, origin at the reference
position (the recording's first pose). Per point it holds the number of observing frames, the triangulation angle and
the estimated distance error, so the obstruction step (#49) can filter, and its colour in the frames.

OpenCV is imported on first use: in the browser it is a 10.7 MB download, loaded only for this step.

Camera conventions: WebXR cameras look along their -z axis with y up (`core.ar_recording`); OpenCV cameras look along
+z with y down. Pixel coordinates have x to the right and y down from the image's top left corner.
"""

import copy
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
from pv_yield_estimator.core.ar_recording import ar_to_earth, frame_rotations, recording_heading, rotation_from_quaternion
from pv_yield_estimator.file_format import FORMAT_VERSION_KEY, check_format_version

log = logging.getLogger(__name__)

POINTS_FORMAT_VERSION = 1
POINTS_KIND = "reconstructed_points"
POINT_COLUMNS = ["east_m", "north_m", "up_m", "n_frames", "triangulation_angle_deg", "distance_m", "distance_error_m", "reprojection_error_px", "red", "green", "blue"]
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
        ### OpenCV's feature positions have pixel centres at whole numbers, the camera matrix at half numbers.
        self.camera_matrices = np.array([camera_matrix(frame["projection_matrix"], width, height) - [[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]] for frame, (width, height) in zip(frames, image_sizes)]).reshape(-1, 3, 3)

    def corrected(self, frames, rotation_vectors, position_offsets):
        """A copy with the given frames' poses corrected: world to camera rotations turned by the rotation vectors (camera axes) and camera centres shifted (AR coordinates, metres)."""
        from scipy.spatial.transform import Rotation
        adjusted = copy.copy(self)
        adjusted.world_to_camera, adjusted.centres = self.world_to_camera.copy(), self.centres.copy()
        adjusted.world_to_camera[frames] = Rotation.from_rotvec(rotation_vectors).as_matrix() @ self.world_to_camera[frames]
        adjusted.centres[frames] = self.centres[frames] + position_offsets
        adjusted.translations = -np.einsum("nij,nj->ni", adjusted.world_to_camera, adjusted.centres)
        adjusted.forward = adjusted.world_to_camera[:, 2, :]
        return adjusted

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


def decode_frame(image_bytes, max_image_px=960, colour=False):
    """Grayscale image (uint8; with `colour` RGB) from JPEG bytes, downscaled so that its longer side is at most `max_image_px` (None: full size)."""
    import cv2
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR if colour else cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Frame image could not be decoded ({len(image_bytes)} bytes, starting with {bytes(image_bytes[:8])!r})")
    scale = 1.0 if max_image_px is None else min(1.0, max_image_px / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if colour else image


def feature_detector(feature="orb", max_features=2000):
    """OpenCV feature detector and descriptor (`orb` or `sift`) and the norm to match its descriptors with."""
    import cv2
    if feature == "orb":
        ### A low corner threshold and border, so low-contrast texture (e.g. a wooden table) gets features too.
        return cv2.ORB_create(nfeatures=max_features, fastThreshold=5, edgeThreshold=15, patchSize=15), cv2.NORM_HAMMING
    if feature == "sift":
        return cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.01), cv2.NORM_L2
    raise ValueError(f"Unknown feature type {feature!r}; known: orb, sift")


def detect_features(image, feature="orb", max_features=2000, feature_grid=(6, 3)):
    """Features of a grayscale image spread over it: local contrast enhanced (CLAHE), then up to `max_features` / cells features per cell of a grid (rows, columns); returns pixel positions (n, 2) and descriptors.

    Without the grid, the strongest corners of the whole image win: in an indoor test, 2000 ORB features left 24 for
    the lower half of the frames, a wooden table.
    """
    import cv2
    rows, columns = feature_grid
    detector, _ = feature_detector(feature, max(1, max_features // (rows * columns)))
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image)
    height, width = image.shape
    keypoints = []
    for row in range(rows):
        for column in range(columns):
            mask = np.zeros_like(image)
            mask[row * height // rows:(row + 1) * height // rows, column * width // columns:(column + 1) * width // columns] = 255
            keypoints += detector.detect(enhanced, mask)
    keypoints, descriptors = detector.compute(enhanced, keypoints)
    return np.array([point.pt for point in keypoints], dtype=float).reshape(-1, 2), descriptors


def select_frame_pairs(poses, usable, fov_deg, min_baseline_m=0.05, max_view_angle_deg=None, max_pairs_per_frame=8):
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


def frame_turn_rates(recording):
    """Turn rate of the camera (degrees per second) at each frame: as recorded (`turn_rate_deg_s`, smoothed over the XR frames), else the mean of the rates to the neighbouring frames.

    Turning blurs the image far more than moving: in a test recording, frames taken while turning faster than about
    20°/s had almost no usable features (#48).
    """
    frames = recording["frames"]
    if not frames:
        return np.zeros(0)
    rotations = frame_rotations(recording)
    times = np.array([frame["time_s"] for frame in frames], dtype=float)
    steps = np.array([np.degrees(np.arccos(np.clip((np.trace(rotations[index].T @ rotations[index - 1]) - 1) / 2, -1.0, 1.0))) for index in range(1, len(frames))])
    step_rates = steps / np.maximum(np.diff(times), 1e-3)
    estimated = np.nanmean(np.stack([np.r_[np.nan, step_rates], np.r_[step_rates, np.nan]]), axis=0) if len(frames) > 1 else np.zeros(1)
    recorded = np.array([frame.get("turn_rate_deg_s", np.nan) for frame in frames], dtype=float)
    return np.where(np.isnan(recorded), estimated, recorded)


def verify_matches(pixels_first, pixels_second, camera_first, camera_second, ransac_px=1.5):
    """Mask of the matches consistent with one relative camera motion, estimated from the images alone (essential matrix, RANSAC with `ransac_px` tolerance); independent of the recorded poses."""
    import cv2
    if len(pixels_first) < 8:
        return np.zeros(len(pixels_first), dtype=bool)
    normalized = [(np.concatenate([pixels, np.ones((len(pixels), 1))], axis=1) @ np.linalg.inv(camera).T)[:, :2] for pixels, camera in ((pixels_first, camera_first), (pixels_second, camera_second))]
    _, mask = cv2.findEssentialMat(normalized[0], normalized[1], focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999, threshold=ransac_px / camera_first[0, 0])
    return np.zeros(len(pixels_first), dtype=bool) if mask is None else mask.ravel().astype(bool)


def reconstruct_points(recording, frame_images, heading_offset_deg=None, feature="orb", max_features=2000, max_image_px=960, feature_grid=(6, 3), max_turn_rate_deg_s=20.0, min_baseline_m=0.05, max_view_angle_deg=None,
                       max_pairs_per_frame=8, ratio=0.8, max_pose_error_px=20.0, ransac_px=1.5, min_pair_matches=20, adjust_poses=True, rotation_prior_deg=1.0, position_prior_m=0.05,
                       max_rotation_correction_deg=3.0, max_position_correction_m=0.3, max_adjustment_points=5000, max_reprojection_px=2.0, min_triangulation_angle_deg=2.0, min_frames=2, min_distance_m=0.1, pixel_error_px=1.0, **kwargs):
    """Point cloud from an AR recording (`recording.json` as dict) and its frames' JPEG bytes (same order); returns `ReconstructedPoints`.

    Settings (config section `sky_obstruction.ar_video`):
    - `heading_offset_deg`: turns the AR coordinates to north; default as in the inspection view (set by hand, else
      from the compass).
    - `feature` (`orb` or `sift`), `max_features` per frame spread over a `feature_grid` (`detect_features`),
      `max_image_px`: frames are downscaled to this longer side (None: full size). Frames with lost tracking (emulated position) are not used, nor those taken while turning
      faster than `max_turn_rate_deg_s` (blurred, `frame_turn_rates`).
    - Frame pairs (`select_frame_pairs`): `min_baseline_m`, `max_view_angle_deg` (default half the field of view),
      `max_pairs_per_frame`. Short baselines are needed for near objects (their view changes quickly); far points
      still get enough parallax as tracks join matches over many frames.
    - Matching: candidates within `max_pose_error_px` of the epipolar line of the recorded poses (generous, as the
      poses may be off by a degree), Lowe's `ratio` test, then checked against the images alone (`verify_matches`,
      `ransac_px`); a pair needs at least `min_pair_matches` matches.
    - With `adjust_poses`, the camera poses are refined from the images (`adjust_frame_poses`, bundle adjustment
      with up to `max_adjustment_points` points), staying close to the recorded ones (`rotation_prior_deg`,
      `position_prior_m`); frames whose observations still don't fit, or that needed a correction beyond
      `max_rotation_correction_deg` or `max_position_correction_m`, are dropped.
    - Points: triangulated per track (matches joined over all frames), observations off by more than
      `max_reprojection_px` dropped, then at least `min_frames` observing frames and a triangulation angle of at
      least `min_triangulation_angle_deg`, at least `min_distance_m` from the observing cameras (on average; rays from
      nearly the same position, e.g. when turning on the spot, meet at the camera). `pixel_error_px` is the assumed
      feature position error for the estimated distance error.
    """
    import cv2
    timings = {}
    start = time.perf_counter()
    frames = recording["frames"]
    if len(frame_images) != len(frames):
        raise ValueError(f"The AR recording has {len(frames)} frames but {len(frame_images)} images were given")
    heading = recording_heading(recording, heading_offset_deg)
    tracked = np.array([not frame.get("emulated_position") for frame in frames], dtype=bool)
    sharp = frame_turn_rates(recording) <= max_turn_rate_deg_s
    usable = tracked & sharp
    colour_images = [decode_frame(image, max_image_px, colour=True) if use else None for image, use in zip(frame_images, usable)]
    images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image is not None else None for image in colour_images]
    image_sizes = [(image.shape[1], image.shape[0]) if image is not None else (frame["width"], frame["height"]) for image, frame in zip(images, frames)]
    poses = FramePoses(recording, image_sizes)
    timings["decode_s"] = time.perf_counter() - start

    start = time.perf_counter()
    _, norm = feature_detector(feature, max_features)
    keypoints, descriptors = [], []
    for image, use in zip(images, usable):
        found, described = detect_features(image, feature, max_features, feature_grid) if use else (np.zeros((0, 2)), None)
        keypoints.append(found)
        descriptors.append(described)
    timings["features_s"] = time.perf_counter() - start

    start = time.perf_counter()
    fov_deg = float(np.median([np.degrees(2 * np.arctan(max(size) / 2 / max(matrix[0, 0], matrix[1, 1]))) for size, matrix in zip(image_sizes, poses.camera_matrices)])) if frames else 60.0
    pairs = select_frame_pairs(poses, usable, fov_deg, min_baseline_m, max_view_angle_deg, max_pairs_per_frame)
    offsets = np.concatenate([[0], np.cumsum([len(points) for points in keypoints])])
    edges, n_matches, pose_disagreement = [], 0, []
    for first, second in pairs:
        fundamental = poses.fundamental_matrix(first, second)
        matches = match_pair(descriptors[first], descriptors[second], norm, ratio, epipolar_band(fundamental, keypoints[first], keypoints[second], max_pose_error_px))
        n_matches += len(matches)
        if len(matches) < min_pair_matches:
            continue
        pixels_first, pixels_second = keypoints[first][matches[:, 0]], keypoints[second][matches[:, 1]]
        consistent = verify_matches(pixels_first, pixels_second, poses.camera_matrices[first], poses.camera_matrices[second], ransac_px)
        if np.count_nonzero(consistent) < min_pair_matches:
            continue
        pose_disagreement.append(float(np.median(epipolar_distances(fundamental, pixels_first[consistent], pixels_second[consistent]))))
        edges.append(np.stack([matches[consistent, 0] + offsets[first], matches[consistent, 1] + offsets[second]], axis=1))
    n_verified_pairs = len(edges)
    edges = np.concatenate(edges) if edges else np.zeros((0, 2), dtype=int)
    timings["matching_s"] = time.perf_counter() - start

    start = time.perf_counter()
    tracks = build_tracks(edges, offsets, keypoints, colour_images)
    adjustment = {}
    if adjust_poses and tracks is not None:
        poses, adjustment, frame_fits = adjust_frame_poses(poses, tracks, rotation_prior_deg, position_prior_m, max_rotation_correction_deg, max_position_correction_m, max_adjustment_points, max_pose_error_px,
                                                           min_triangulation_angle_deg, max_reprojection_px, pixel_error_px)
        usable &= frame_fits
    timings["adjustment_s"] = time.perf_counter() - start

    start = time.perf_counter()
    points, observation_count = points_from_tracks(tracks, poses, usable, max_reprojection_px, min_triangulation_angle_deg, min_frames, min_distance_m, pixel_error_px)
    to_earth = ar_to_earth(heading["heading_offset_deg"])
    points[["east_m", "north_m", "up_m"]] = points[["east_m", "north_m", "up_m"]].to_numpy() @ to_earth.T
    timings["triangulation_s"] = time.perf_counter() - start

    statistics = {"n_frames": len(frames), "n_frames_tracking_lost": int((~tracked).sum()), "n_frames_blurred": int((tracked & ~sharp).sum()), "n_frames_used": int(usable.sum()),
                  "n_features": int(offsets[-1]), "n_pairs": len(pairs), "n_pairs_verified": n_verified_pairs, "n_matches": n_matches, "n_verified_matches": len(edges),
                  "pose_disagreement_px": round(float(np.median(pose_disagreement)), 2) if pose_disagreement else None, **adjustment,
                  "n_observations": observation_count, "n_points": len(points), **{key: round(value, 3) for key, value in timings.items()}}
    settings = {"feature": feature, "max_features": max_features, "feature_grid": list(feature_grid), "max_image_px": max_image_px, "max_turn_rate_deg_s": max_turn_rate_deg_s, "min_baseline_m": min_baseline_m, "max_view_angle_deg": max_view_angle_deg,
                "max_pairs_per_frame": max_pairs_per_frame, "ratio": ratio, "max_pose_error_px": max_pose_error_px, "ransac_px": ransac_px, "min_pair_matches": min_pair_matches, "adjust_poses": adjust_poses,
                "rotation_prior_deg": rotation_prior_deg, "position_prior_m": position_prior_m,
                "max_rotation_correction_deg": max_rotation_correction_deg, "max_position_correction_m": max_position_correction_m, "max_adjustment_points": max_adjustment_points, "max_reprojection_px": max_reprojection_px,
                "min_triangulation_angle_deg": min_triangulation_angle_deg, "min_frames": min_frames, "min_distance_m": min_distance_m, "pixel_error_px": pixel_error_px}
    metadata = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "program_version": __version__, "recording_created": recording.get("created"), **heading,
                "settings": settings, "statistics": statistics}
    log.info(f"Point cloud from the AR recording: {len(points)} points from {statistics['n_frames_used']} of {len(frames)} frames ({statistics['n_frames_blurred']} blurred), "
             f"{n_verified_pairs} of {len(pairs)} frame pairs verified, recorded poses off by {statistics['pose_disagreement_px']} px (median)"
             + (f", adjusted by {adjustment['rotation_correction_deg_median']:.2f}° / {100 * adjustment['position_correction_m_median']:.1f} cm (median)" if adjustment else "")
             + f"; {sum(timings.values()):.1f} s (" + ", ".join(f"{key[:-2]} {value:.1f} s" for key, value in timings.items()) + ")")
    return ReconstructedPoints(points, metadata, poses.centres @ to_earth.T)


def depth_map(depth):
    """Depth map (metres, rows × columns of the depth buffer; 0 where unknown) from a frame's `depth` entry as recorded (base64 data in `format`: luminance-alpha or unsigned-short as 16-bit integers, or float32)."""
    import base64
    data = base64.b64decode(depth["data_base64"])
    dtype = {"luminance-alpha": "<u2", "unsigned-short": "<u2", "float32": "<f4"}.get(depth["format"])
    if dtype is None:
        raise ValueError(f"Unknown depth data format {depth['format']!r}; known: luminance-alpha, unsigned-short, float32")
    values = np.frombuffer(data, dtype=dtype).astype(float)
    if len(values) != depth["width"] * depth["height"]:
        raise ValueError(f"Depth map of {len(values)} values, expected {depth['width']} x {depth['height']}")
    return values.reshape(depth["height"], depth["width"]) * depth["raw_value_to_meters"]


def depth_map_points(frame, max_depth_m=5.0):
    """Points (AR coordinates) of a frame's depth map with the normalized view coordinates (x right, y down, 0–1) of their pixels.

    Each depth buffer pixel centre is taken back to normalized view coordinates with the inverse of the recorded
    `norm_depth_buffer_from_norm_view` transform, then along the camera ray (projection matrix) to its depth, taken as
    the distance along the viewing direction (ARCore's depth maps), and into AR coordinates with the frame's pose.
    """
    depth = frame["depth"]
    distances = depth_map(depth)
    rows, columns = np.indices(distances.shape)
    valid = (distances > 0) & (distances <= max_depth_m)
    buffer_coordinates = np.stack([(columns[valid] + 0.5) / depth["width"], (rows[valid] + 0.5) / depth["height"], np.zeros(np.count_nonzero(valid)), np.ones(np.count_nonzero(valid))])
    view_coordinates = np.linalg.inv(np.asarray(depth["norm_depth_buffer_from_norm_view"], dtype=float).reshape(4, 4).T) @ buffer_coordinates
    view_x, view_y = view_coordinates[0] / view_coordinates[3], view_coordinates[1] / view_coordinates[3]
    projection = np.asarray(frame["projection_matrix"], dtype=float)
    ### Normalized device coordinates, then the camera ray (WebXR camera axes: x right, y up, looking along -z) at depth 1.
    right = (2 * view_x - 1 + projection[8]) / projection[0]
    up = (1 - 2 * view_y + projection[9]) / projection[5]
    in_camera = np.stack([right, up, -np.ones_like(right)], axis=1) * distances[valid][:, None]
    rotation = rotation_from_quaternion(frame["orientation_xyzw"])[0]
    return in_camera @ rotation.T + np.asarray(frame["position_m"], dtype=float), np.stack([view_x, view_y], axis=1)


def points_from_depth_maps(recording, frame_images, heading_offset_deg=None, max_depth_m=5.0, voxel_m=0.03, min_depth_frames=2, max_turn_rate_deg_s=20.0, **kwargs):
    """Dense point cloud from the depth maps recorded with the frames (ARCore's Depth API via WebXR depth sensing); returns `ReconstructedPoints`.

    Settings (config section `sky_obstruction.ar_video`): depths beyond `max_depth_m` are dropped (ARCore's depth is
    most accurate within a few metres); the points of all frames are merged into cubes of `voxel_m` (mean position and
    colour), keeping cubes seen in at least `min_depth_frames` frames. Frames with lost tracking or taken while turning
    faster than `max_turn_rate_deg_s` are not used. The recorded poses are used as they are.
    """
    start = time.perf_counter()
    frames = recording["frames"]
    heading = recording_heading(recording, heading_offset_deg)
    usable = np.array([bool(frame.get("depth")) and not frame.get("emulated_position") for frame in frames]) & (frame_turn_rates(recording) <= max_turn_rate_deg_s)
    all_points, all_colours, all_frames, all_distances = [], [], [], []
    for index in np.flatnonzero(usable):
        frame = frames[index]
        points, view_coordinates = depth_map_points(frame, max_depth_m)
        image = decode_frame(frame_images[index], None, colour=True)
        columns = np.clip((view_coordinates[:, 0] * image.shape[1]).astype(int), 0, image.shape[1] - 1)
        rows = np.clip((view_coordinates[:, 1] * image.shape[0]).astype(int), 0, image.shape[0] - 1)
        all_points.append(points)
        all_colours.append(image[rows, columns].astype(float))
        all_frames.append(np.full(len(points), index))
        all_distances.append(np.linalg.norm(points - np.asarray(frame["position_m"], dtype=float), axis=1))
    if not all_points:
        table = pd.DataFrame(columns=POINT_COLUMNS, dtype=float)
    else:
        points, colours, frame_ids, distances = (np.concatenate(values) for values in (all_points, all_colours, all_frames, all_distances))
        _, cell, counts = np.unique(np.floor(points / voxel_m).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
        cell = cell.ravel()
        n_cells = len(counts)
        frames_per_cell = np.bincount(np.unique(np.stack([cell, frame_ids], axis=1), axis=0)[:, 0], minlength=n_cells)
        mean = lambda values: np.stack([np.bincount(cell, weights=values[:, axis], minlength=n_cells) for axis in range(values.shape[1])], axis=1) / counts[:, None]
        positions, cell_colours, cell_distances = mean(points), mean(colours), mean(distances[:, None])[:, 0]
        keep = frames_per_cell >= min_depth_frames
        positions = positions[keep] @ ar_to_earth(heading["heading_offset_deg"]).T
        table = pd.DataFrame({"east_m": positions[:, 0], "north_m": positions[:, 1], "up_m": positions[:, 2], "n_frames": frames_per_cell[keep], "triangulation_angle_deg": np.nan,
                              "distance_m": cell_distances[keep], "distance_error_m": np.nan, "reprojection_error_px": np.nan,
                              **{name: np.round(cell_colours[keep, channel]).astype(int) for channel, name in enumerate(("red", "green", "blue"))}})
    statistics = {"n_frames": len(frames), "n_frames_with_depth": sum(bool(frame.get("depth")) for frame in frames), "n_frames_used": int(usable.sum()), "n_points": len(table),
                  "depth_s": round(time.perf_counter() - start, 3)}
    settings = {"points_from": "depth", "max_depth_m": max_depth_m, "voxel_m": voxel_m, "min_depth_frames": min_depth_frames, "max_turn_rate_deg_s": max_turn_rate_deg_s}
    metadata = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "program_version": __version__, "recording_created": recording.get("created"), **heading,
                "settings": settings, "statistics": statistics}
    log.info(f"Point cloud from the depth maps: {len(table)} cubes of {100 * voxel_m:g} cm from {statistics['n_frames_used']} of {len(frames)} frames ({statistics['n_frames_with_depth']} with depth) in {statistics['depth_s']:.1f} s")
    positions = np.array([frame["position_m"] for frame in frames], dtype=float).reshape(-1, 3)
    return ReconstructedPoints(table, metadata, positions @ ar_to_earth(heading["heading_offset_deg"]).T)


def recording_points(recording, frame_images, points_from="auto", **settings):
    """Point cloud from an AR recording: from the recorded depth maps (`points_from_depth_maps`) or by matching and triangulating features (`reconstruct_points`); `auto` uses the depth maps if the recording has any."""
    if points_from not in ("auto", "depth", "features"):
        raise ValueError(f"Unknown points_from {points_from!r}; known: auto, depth, features")
    has_depth = any(frame.get("depth") for frame in recording["frames"])
    if points_from == "depth" and not has_depth:
        raise ValueError("The AR recording has no depth maps (the phone or browser offered no depth sensing); use points_from: features")
    if points_from == "depth" or (points_from == "auto" and has_depth):
        return points_from_depth_maps(recording, frame_images, **settings)
    return reconstruct_points(recording, frame_images, **settings)


class Tracks:
    """Features matched over several frames (connected components of the verified matches): per observation the frame, the pixel, its colour (RGB) and the track it belongs to."""

    def __init__(self, frame_ids, pixels, track_ids, colours):
        self.frame_ids, self.pixels, self.track_ids, self.colours = frame_ids, pixels, track_ids, colours
        self.n_tracks = int(track_ids.max()) + 1 if len(track_ids) else 0


def build_tracks(edges, offsets, keypoints, colour_images):
    """Join the verified matches (pairs of global feature numbers) into tracks, with each observation's colour from its frame (RGB images); None without matches."""
    if len(edges) == 0:
        return None
    n_features = offsets[-1]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_features, n_features))
    _, labels = connected_components(graph, directed=False)
    features = np.unique(edges)
    _, track_ids = np.unique(labels[features], return_inverse=True)
    frame_ids = np.searchsorted(offsets, features, side="right") - 1
    pixels = np.concatenate(keypoints)[features]
    colours = np.zeros((len(features), 3))
    for frame in np.unique(frame_ids):
        here = frame_ids == frame
        image = colour_images[frame]
        rows, columns = (np.clip(np.round(pixels[here, axis]).astype(int), 0, image.shape[1 - axis] - 1) for axis in (1, 0))
        colours[here] = image[rows, columns]
    return Tracks(frame_ids, pixels, track_ids, colours)


def triangulate_observations(tracks, poses, keep, max_reprojection_px=2.0, rounds=2):
    """Triangulate the tracks from the kept observations, drop those off by more than `max_reprojection_px` or behind the camera (`rounds` times), then triangulate the rest; returns points (AR coordinates), the kept mask and the reprojection errors."""
    rays = poses.rays(tracks.frame_ids, tracks.pixels)
    centres = poses.centres[tracks.frame_ids]
    keep = keep.copy()
    for final in [False] * rounds + [True]:
        points = triangulate_tracks(tracks.track_ids[keep], centres[keep], rays[keep], tracks.n_tracks)
        projected, depth = poses.project(tracks.frame_ids, points[tracks.track_ids])
        errors = np.linalg.norm(projected - tracks.pixels, axis=1)
        if not final:
            keep &= (errors <= max_reprojection_px) & (depth > 0)
    return points, keep, errors


def adjust_frame_poses(poses, tracks, rotation_prior_deg=1.0, position_prior_m=0.05, max_rotation_correction_deg=3.0, max_position_correction_m=0.3, max_adjustment_points=5000, max_pose_error_px=20.0, min_triangulation_angle_deg=2.0, max_reprojection_px=2.0, pixel_error_px=1.0):
    """Refine the recorded camera poses from the images (bundle adjustment); returns the adjusted poses, statistics and the mask of frames that fit.

    The tracks are triangulated with the recorded poses (observations off by up to `max_pose_error_px`), and the
    best-observed points with enough parallax (at most `max_adjustment_points`) are adjusted together with a small
    rotation and shift per frame, minimizing the reprojection errors (robust, soft-L1) plus priors that keep each frame
    within about `rotation_prior_deg` and `position_prior_m` of its recorded pose; the priors also keep the recorded
    scale, vertical and heading. A frame fits if its median reprojection error afterwards is at most
    `max_reprojection_px` and its correction stays within `max_rotation_correction_deg` and
    `max_position_correction_m` (larger ones mean its recorded pose or its matches are wrong, so neither is trusted);
    frames that don't fit keep their recorded pose.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    points, keep, _ = triangulate_observations(tracks, poses, np.ones(len(tracks.track_ids), dtype=bool), max_pose_error_px)
    to_cameras = poses.centres[tracks.frame_ids] - points[tracks.track_ids]
    distances = np.linalg.norm(to_cameras, axis=1)
    angles = track_angles(tracks.track_ids[keep], (to_cameras / np.maximum(distances, 1e-12)[:, None])[keep], tracks.n_tracks)
    counts = np.bincount(tracks.track_ids[keep], minlength=tracks.n_tracks)
    candidates = np.flatnonzero((counts >= 2) & (angles >= min_triangulation_angle_deg / 2))
    chosen = candidates[np.argsort(-counts[candidates], kind="stable")[:max_adjustment_points]]
    observed = keep & np.isin(tracks.track_ids, chosen)
    fits = np.ones(len(poses.centres), dtype=bool)
    if len(chosen) < 10:
        log.info(f"Pose adjustment skipped: only {len(chosen)} points with enough parallax")
        return poses, {}, fits
    frames, frame_index = np.unique(tracks.frame_ids[observed], return_inverse=True)
    chosen_tracks, point_index = np.unique(tracks.track_ids[observed], return_inverse=True)
    observed_pixels = tracks.pixels[observed]
    n_frames, n_points, n_observations = len(frames), len(chosen_tracks), len(observed_pixels)
    recorded_rotations, recorded_centres, cameras = poses.world_to_camera[frames], poses.centres[frames], poses.camera_matrices[frames]
    rotation_scale = np.radians(rotation_prior_deg)
    robust_scale = 3.0

    def robust(errors):
        """Pseudo-residuals whose squares give the soft-L1 loss: quadratic for small errors, growing like the square root of large ones, so wrong matches don't pull.

        Applied to the reprojection errors only: `least_squares`' own loss would also soften the priors and let single
        frames jump by metres.
        """
        scaled = errors / robust_scale
        return robust_scale * np.sign(scaled) * np.sqrt(2.0 * (np.sqrt(1.0 + scaled * scaled) - 1.0))

    def residuals(parameters):
        """Robust reprojection errors (in units of `pixel_error_px`) and the priors of the frame corrections."""
        corrections = parameters[:6 * n_frames].reshape(n_frames, 6)
        positions = parameters[6 * n_frames:].reshape(n_points, 3)
        rotations = Rotation.from_rotvec(corrections[:, :3]).as_matrix() @ recorded_rotations
        in_camera = np.einsum("nij,nj->ni", rotations[frame_index], positions[point_index] - (recorded_centres + corrections[:, 3:])[frame_index])
        homogeneous = np.einsum("nij,nj->ni", cameras[frame_index], in_camera)
        projected = homogeneous[:, :2] / np.maximum(homogeneous[:, 2], 1e-3)[:, None]
        return np.concatenate([robust((projected - observed_pixels) / pixel_error_px).ravel(), (corrections[:, :3] / rotation_scale).ravel(), (corrections[:, 3:] / position_prior_m).ravel()])

    ### Which residual depends on which parameter: each pixel coordinate on its frame's 6 corrections and its point's 3 coordinates, each prior on its correction.
    rows = (2 * np.arange(n_observations)[:, None] + np.arange(2))[:, :, None]
    columns = np.concatenate([6 * frame_index[:, None] + np.arange(6), 6 * n_frames + 3 * point_index[:, None] + np.arange(3)], axis=1)[:, None, :]
    rows, columns = np.broadcast_arrays(rows, columns)
    rows = np.concatenate([rows.ravel(), 2 * n_observations + np.arange(6 * n_frames)])
    columns = np.concatenate([columns.ravel(), np.arange(6 * n_frames)])
    sparsity = coo_matrix((np.ones(len(rows), dtype=int), (rows, columns)), shape=(2 * n_observations + 6 * n_frames, 6 * n_frames + 3 * n_points))
    start = np.concatenate([np.zeros(6 * n_frames), points[chosen_tracks].ravel()])
    result = least_squares(residuals, start, jac_sparsity=sparsity, x_scale="jac", method="trf", max_nfev=50)
    corrections = result.x[:6 * n_frames].reshape(n_frames, 6)
    rotation_deg, position_m = np.degrees(np.linalg.norm(corrections[:, :3], axis=1)), np.linalg.norm(corrections[:, 3:], axis=1)
    adjusted = poses.corrected(frames, corrections[:, :3], corrections[:, 3:])
    in_camera = np.einsum("nij,nj->ni", adjusted.world_to_camera[frames][frame_index], result.x[6 * n_frames:].reshape(n_points, 3)[point_index] - adjusted.centres[frames][frame_index])
    homogeneous = np.einsum("nij,nj->ni", cameras[frame_index], in_camera)
    errors = np.linalg.norm(homogeneous[:, :2] / np.maximum(homogeneous[:, 2], 1e-3)[:, None] - observed_pixels, axis=1)
    frame_errors = np.array([np.median(errors[frame_index == index]) for index in range(n_frames)])
    rejected = (frame_errors > max_reprojection_px) | (rotation_deg > max_rotation_correction_deg) | (position_m > max_position_correction_m)
    fits[frames[rejected]] = False
    adjusted = poses.corrected(frames[~rejected], corrections[~rejected, :3], corrections[~rejected, 3:])
    statistics = {"adjusted_frames": n_frames, "adjusted_points": n_points, "rotation_correction_deg_median": round(float(np.median(rotation_deg)), 3), "rotation_correction_deg_max": round(float(rotation_deg.max()), 3),
                  "position_correction_m_median": round(float(np.median(position_m)), 4), "position_correction_m_max": round(float(position_m.max()), 4),
                  "reprojection_px_median_after_adjustment": round(float(np.median(errors)), 3), "n_frames_not_fitting": int(np.count_nonzero(rejected))}
    log.debug(f"Pose adjustment: {statistics}, {result.nfev} evaluations, {result.message}")
    return adjusted, statistics, fits


def points_from_tracks(tracks, poses, usable, max_reprojection_px=2.0, min_triangulation_angle_deg=2.0, min_frames=2, min_distance_m=0.1, pixel_error_px=1.0):
    """Triangulate the tracks' observations in usable frames and keep the reliable points; returns the points table (AR coordinates) and the number of observations used."""
    if tracks is None:
        return pd.DataFrame(columns=POINT_COLUMNS, dtype=float), 0
    n_tracks = tracks.n_tracks
    points, keep, errors = triangulate_observations(tracks, poses, usable[tracks.frame_ids], max_reprojection_px)
    track_ids, frame_ids = tracks.track_ids, tracks.frame_ids
    ### Distinct observing frames per track; a track seen twice in a frame counts that frame once.
    track_frames = np.unique(np.stack([track_ids[keep], frame_ids[keep]], axis=1), axis=0)
    n_frames = np.bincount(track_frames[:, 0], minlength=n_tracks)
    to_cameras = poses.centres[frame_ids[keep]] - points[track_ids[keep]]
    distances = np.linalg.norm(to_cameras, axis=1)
    angles = track_angles(track_ids[keep], to_cameras / np.maximum(distances, 1e-12)[:, None], n_tracks)
    observations = np.maximum(np.bincount(track_ids[keep], minlength=n_tracks), 1)
    mean_distance = np.bincount(track_ids[keep], weights=distances, minlength=n_tracks) / observations
    max_error = np.zeros(n_tracks)
    np.maximum.at(max_error, track_ids[keep], errors[keep])
    focal_px = np.median(poses.camera_matrices[:, 0, 0])
    distance_error = mean_distance * (pixel_error_px / focal_px) / np.maximum(np.sin(np.radians(angles)), 1e-12)
    valid = (n_frames >= min_frames) & (angles >= min_triangulation_angle_deg) & (mean_distance >= min_distance_m)
    colours = np.stack([np.bincount(track_ids[keep], weights=tracks.colours[keep, channel], minlength=n_tracks) for channel in range(3)], axis=1) / observations[:, None]
    table = pd.DataFrame({"east_m": points[valid, 0], "north_m": points[valid, 1], "up_m": points[valid, 2], "n_frames": n_frames[valid], "triangulation_angle_deg": angles[valid],
                          "distance_m": mean_distance[valid], "distance_error_m": distance_error[valid], "reprojection_error_px": max_error[valid],
                          **{name: np.round(colours[valid, channel]).astype(int) for channel, name in enumerate(("red", "green", "blue"))}})
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
