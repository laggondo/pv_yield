#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PV Shading Analysis from LiDAR Point Cloud

Calculates the annual irradiation on a PV panel considering shading from
surrounding structures measured with a Livox Mid-360 LiDAR.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.transform import Rotation

from meteo_data_handler import MeteoDataHandlerTMY3


# =============================================================================
# Data Loading
# =============================================================================

def load_point_cloud(filepath, max_points=None):
    """Load Livox Mid-360 CSV point cloud, filter out invalid (0,0,0) points
    and points below the horizontal plane (z <= 0).

    Args:
        filepath: Path to CSV file
        max_points: If set, randomly subsample (without replacement) to this
            many points after filtering. Uses uniform random selection to
            ensure even coverage of the non-repetitive scan pattern.

    Returns: np.ndarray of shape (N, 3) with columns [x, y, z]
    """
    df = pd.read_csv(filepath, header=0, skiprows=[1], usecols=['X', 'Y', 'Z', 'Reflectivity'])

    # Filter out no-return points
    df = df[df['Reflectivity'] > 0]

    points = df[['X', 'Y', 'Z']].values.astype(float)

    mask_nonzero = ~np.all(np.isclose(points, 0.0), axis=1)
    points = points[mask_nonzero]

    mask_above = points[:, 2] > 0.0
    points = points[mask_above]

    # Random subsample if requested
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng()
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]

    return points

# =============================================================================
# Coordinate Transformations
# =============================================================================

def filter_upper_hemisphere(points):
    """Remove points at or below the horizontal plane (z <= 0) and zero-length
    vectors. Intended to be called after apply_offset, since the offset may
    move some structures below the evaluation plane.

    Args:
        points: (N, 3) array

    Returns: (M, 3) filtered array with M <= N
    """
    norms = np.linalg.norm(points, axis=1)
    mask = (points[:, 2] > 0.0) & (norms > 0.0)
    return points[mask]


def points_to_spherical(points):
    """Convert Cartesian points to spherical coordinates (zenith, azimuth)
    following the Duffie-Beckman convention via zen_azm_from_vec.

    zen_azm_from_vec expects vectors pointing FROM the sky object TOWARDS
    the observer (i.e. downward-pointing solar vectors). Our point cloud
    vectors point FROM the sensor TOWARDS the structures (upward), so we
    negate them before passing.

    Args:
        points: (N, 3) array in Duffie-Beckman frame (upper hemisphere only)

    Returns: (zen, azm) each of shape (N,) in radians
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    directions = points / norms
    zen, azm = ul.zen_azm_from_vec(-directions)
    return zen, azm


# =============================================================================
# Sky Discretization & Obstruction
# =============================================================================

def build_sky_discretization(n_nodes=500):
    """Create a DensitySkyDiscretization for the upper hemisphere and
    compute its convex-hull triangulation.

    Args:
        n_nodes: Approximate number of sky patches

    Returns: DensitySkyDiscretization instance (with triangulation)
    """
    sky_disc = ul.DensitySkyDiscretization(n_nodes_estim=n_nodes)
    sky_disc.create_triangulation()
    return sky_disc

def compute_obstruction_mask(sky_disc, point_zen, point_azm, min_points_per_patch=1):
    """Determine which sky patches (simplices) are obstructed by at least
    min_points_per_patch LiDAR points.

    Uses KDTree bulk query to find candidate simplices per point, then checks
    barycentric coordinates in vectorized batches. A simplex is only marked
    as blocked if the number of points falling inside it meets the threshold.

    Args:
        sky_disc: DensitySkyDiscretization instance (with triangulation)
        point_zen: (N,) zenith angles of point cloud [rad]
        point_azm: (N,) azimuth angles of point cloud [rad]
        min_points_per_patch: Minimum number of points required to mark a
            patch as blocked. Higher values reject noise at the cost of
            potentially missing thin structures.

    Returns: boolean array of shape (n_simplices,), True = blocked
    """
    point_vecs = ul.vec_from_zen_azm(point_zen, point_azm)

    n_simplices = len(sky_disc.simplices)
    point_counts = np.zeros(n_simplices, dtype=int)

    # Build lookup: for each node, which simplices contain it?
    node_to_simplices = [[] for _ in range(len(sky_disc.nodes))]
    for si, simplex in enumerate(sky_disc.simplices):
        for ni in simplex:
            node_to_simplices[ni].append(si)

    # Precompute inverse matrices for all simplices
    nodes_matrices = sky_disc.nodes[sky_disc.simplices]  # (n_simplices, 3, 3)
    inv_matrices = np.linalg.inv(np.transpose(nodes_matrices, (0, 2, 1)))  # (n_simplices, 3, 3)

    # Bulk KDTree query: find k=3 nearest nodes per point
    _, nearest_indices = sky_disc.kd_tree.query(point_vecs, k=3)

    # For each point, check only candidate simplices
    for pi in range(len(point_vecs)):
        # Gather candidate simplices from nearest nodes
        candidate_set = set()
        for ni in nearest_indices[pi]:
            candidate_set.update(node_to_simplices[ni])
        candidates = np.array(list(candidate_set))
        if len(candidates) == 0:
            continue

        # Barycentric check for candidates only
        coords = inv_matrices[candidates] @ point_vecs[pi]  # (n_cand, 3)
        inside = np.all(coords >= -1e-10, axis=1)
        point_counts[candidates[inside]] += 1

    mask = point_counts >= min_points_per_patch
    return mask

def is_sun_blocked(sky_disc, obstruction_mask, sun_vec):
    """Check whether the sun direction falls inside a blocked sky patch.

    Args:
        sky_disc: DensitySkyDiscretization instance
        obstruction_mask: boolean array over simplices, True = blocked
        sun_vec: (3,) unit vector pointing towards the sun (Duffie-Beckman)

    Returns: True if the sun is blocked by an obstruction
    """
    simplex_ind = sky_disc.find_simplex(sun_vec[0], sun_vec[1], sun_vec[2])
    if simplex_ind == -1:
        return False
    return obstruction_mask[simplex_ind]

# =============================================================================
# PV Panel & Irradiance
# =============================================================================

def compute_panel_normal(tilt_deg, azimuth_deg):
    """Compute the outward-facing normal vector of the PV panel.

    Uses Duffie-Beckman convention: azimuth 0 = South, positive towards West.
    Tilt 0 = horizontal (normal pointing up). The normal is simply the unit
    vector at zenith=tilt and azimuth=panel_azimuth.

    Args:
        tilt_deg: Panel tilt angle from horizontal [degrees]
        azimuth_deg: Panel azimuth angle [degrees]

    Returns: (3,) unit normal vector
    """
    tilt_rad = np.radians(tilt_deg)
    azimuth_rad = np.radians(azimuth_deg)
    return ul.vec_from_zen_azm(tilt_rad, azimuth_rad)


def compute_cos_incidence(sun_zen, sun_azm, panel_normal):
    """Compute cosine of the angle of incidence between the sun and the panel.

    The sun direction vector (from surface towards sun) is computed via
    vec_from_zen_azm, then dotted with the panel normal.

    Args:
        sun_zen: Solar zenith angle(s) [rad], scalar or array
        sun_azm: Solar azimuth angle(s) [rad], scalar or array
        panel_normal: (3,) panel normal vector

    Returns: cos(theta_incidence), scalar or array, clamped to >= 0
    """
    sun_vec = ul.vec_from_zen_azm(sun_zen, sun_azm)
    cos_inc = np.dot(sun_vec, panel_normal)
    return np.maximum(cos_inc, 0.0)

def compute_simplex_solid_angles(sky_disc):
    """Compute the solid angle of each simplex (spherical triangle) in the
    sky discretization.

    Uses the formula: Ω = 2 * arctan(|a · (b × c)| / (1 + a·b + b·c + a·c))
    where a, b, c are unit vectors at the triangle vertices.

    Args:
        sky_disc: DensitySkyDiscretization instance

    Returns: (n_simplices,) array of solid angles [sr]
    """
    a = sky_disc.nodes[sky_disc.simplices[:, 0]]
    b = sky_disc.nodes[sky_disc.simplices[:, 1]]
    c = sky_disc.nodes[sky_disc.simplices[:, 2]]

    numerator = np.abs(np.sum(a * np.cross(b, c), axis=1))
    denominator = 1.0 + np.sum(a * b, axis=1) + np.sum(b * c, axis=1) + np.sum(a * c, axis=1)

    solid_angles = 2.0 * np.arctan2(numerator, denominator)
    return solid_angles

def compute_sky_view_factor(sky_disc, obstruction_mask):
    """Compute the sky view factor (SVF) for a horizontal surface, considering
    obstructed patches.

    SVF is the fraction of diffuse irradiance reaching the surface compared to
    an unobstructed hemisphere, assuming isotropic sky radiance. Each patch is
    weighted by cos(zenith) of its centroid (contribution to a horizontal surface).

    Args:
        sky_disc: DensitySkyDiscretization instance
        obstruction_mask: boolean array (n_simplices,), True = blocked

    Returns: float, sky view factor in [0, 1]
    """
    solid_angles = compute_simplex_solid_angles(sky_disc)

    # Compute centroid zenith for each simplex
    a = sky_disc.nodes[sky_disc.simplices[:, 0]]
    b = sky_disc.nodes[sky_disc.simplices[:, 1]]
    c = sky_disc.nodes[sky_disc.simplices[:, 2]]
    centroids = (a + b + c) / 3.0
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    # cos(zenith) = z-component of unit centroid vector
    cos_zen = centroids[:, 2]

    # Weighted contributions
    weights = solid_angles * cos_zen
    total_weight = np.sum(weights)
    unblocked_weight = np.sum(weights[~obstruction_mask])

    svf = unblocked_weight / total_weight
    return svf

def compute_hourly_irradiance(meteo_handler, panel_normal, obstruction_mask, sky_disc, n_sub_steps=6):
    """Compute hourly irradiance on the PV panel for all 8760 hours.

    Beam component:
        - Unobstructed: DNI * cos(theta_incidence), 0 if sun below horizon
        - Obstructed: DNI * cos(theta_incidence) * fraction_unblocked, where
          fraction_unblocked is determined by checking n_sub_steps evenly
          spaced sun positions within each hour.

    Diffuse component:
        - Unobstructed: DHI from TMY file
        - Obstructed: DHI * sky_view_factor (isotropic sky model)

    Args:
        meteo_handler: MeteoDataHandlerTMY3 instance
        panel_normal: (3,) panel normal vector
        obstruction_mask: boolean array (n_simplices,), True = blocked
        sky_disc: DensitySkyDiscretization instance
        n_sub_steps: Number of sub-steps per hour for sun position evaluation.
            Higher values give smoother results but take longer.

    Returns: dict with keys:
        'beam_unobstructed': (8760,) array [Wh/m²]
        'beam_obstructed':   (8760,) array [Wh/m²]
        'diffuse_unobstructed': (8760,) array [Wh/m²]
        'diffuse_obstructed':   (8760,) array [Wh/m²]
        'total_unobstructed': (8760,) array [Wh/m²]
        'total_obstructed':   (8760,) array [Wh/m²]
        'sun_zen': (8760,) array [rad]
        'sun_azm': (8760,) array [rad]
        'timestamps': DatetimeIndex
        'sky_view_factor': float
    """
    sun_zen, sun_azm = meteo_handler.get_zen_azm()
    dni = meteo_handler.df_weather_data['dni'].values.astype(float)
    dhi = meteo_handler.df_weather_data['dhi'].values.astype(float)
    timestamps = meteo_handler.df_weather_data.index
    n_hours = len(dni)

    # Vectorized computation of beam on tilted plane
    cos_inc = compute_cos_incidence(sun_zen, sun_azm, panel_normal)
    sun_above_horizon = sun_zen < np.pi / 2.0
    beam_unobstructed = np.where(sun_above_horizon, dni * cos_inc, 0.0)

    # Compute sub-step sun positions for blocking evaluation
    sp = meteo_handler.get_SunPosition_object()
    timedelta_hour = timestamps[1] - timestamps[0]
    sub_offsets = np.linspace(-0.5, 0.5, n_sub_steps, endpoint=False) + 0.5 / n_sub_steps

    # Determine blocking fraction for each hour
    fraction_unblocked = np.ones(n_hours)
    for i in range(n_hours):
        if not sun_above_horizon[i] or beam_unobstructed[i] <= 0.0:
            continue

        n_blocked = 0
        for offset in sub_offsets:
            dt_sub = timestamps[i] - timedelta_hour * (0.5 - offset)
            zen_sub, azm_sub = sp.get_azm_zen_angles(dt_sub)
            if zen_sub >= np.pi / 2.0:
                continue
            sun_vec_sub = ul.vec_from_zen_azm(zen_sub, azm_sub)
            if is_sun_blocked(sky_disc, obstruction_mask, sun_vec_sub):
                n_blocked += 1

        fraction_unblocked[i] = 1.0 - n_blocked / n_sub_steps

    beam_obstructed = beam_unobstructed * fraction_unblocked

    # Diffuse component with sky view factor correction
    svf = compute_sky_view_factor(sky_disc, obstruction_mask)
    diffuse_unobstructed = dhi.copy()
    diffuse_obstructed = dhi * svf

    return {
        'beam_unobstructed': beam_unobstructed,
        'beam_obstructed': beam_obstructed,
        'diffuse_unobstructed': diffuse_unobstructed,
        'diffuse_obstructed': diffuse_obstructed,
        'total_unobstructed': beam_unobstructed + diffuse_unobstructed,
        'total_obstructed': beam_obstructed + diffuse_obstructed,
        'sun_zen': sun_zen,
        'sun_azm': sun_azm,
        'timestamps': timestamps,
        'sky_view_factor': svf,
    }

# =============================================================================
# Plotting
# =============================================================================

def plot_sky_dome(sky_disc, obstruction_mask, meteo_handler=None, save_path=None):
    """Plot a polar sky-dome showing blocked (red) and free (blue) patches.

    Optionally overlays the annual sun path by highlighting simplices that the
    sun traverses during the year, computed at sub-hourly resolution to avoid
    gaps.

    Args:
        sky_disc: DensitySkyDiscretization instance
        obstruction_mask: boolean array (n_simplices,), True = blocked
        meteo_handler: MeteoDataHandlerTMY3 instance (optional). If provided,
            the annual sun path is overlaid.
        save_path: If provided, save figure to this path.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('S')

    node_zen_deg = np.degrees(sky_disc.sun_positions[:, 0])
    node_azm = sky_disc.sun_positions[:, 1]

    cmap = mcolors.ListedColormap(['skyblue', 'salmon'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.tripcolor(
        node_azm, node_zen_deg, sky_disc.simplices,
        facecolors=obstruction_mask.astype(float),
        cmap=cmap, norm=norm)

    ax.triplot(node_azm, node_zen_deg, sky_disc.simplices, '-', color='grey', linewidth=0.2)

    # Overlay annual sun path at sub-hourly resolution
    if meteo_handler is not None:
        sp = meteo_handler.get_SunPosition_object()
        timestamps = meteo_handler.df_weather_data.index
        timedelta_hour = timestamps[1] - timestamps[0]

        # Compute sun positions every 10 minutes across the year
        sub_steps_per_hour = 6
        sub_offsets = np.linspace(0.0, 1.0, sub_steps_per_hour, endpoint=False)

        sun_simplices = set()
        for i in range(len(timestamps)):
            for offset in sub_offsets:
                dt_sub = timestamps[i] - timedelta_hour * (0.5 - offset)
                zen_sub, azm_sub = sp.get_azm_zen_angles(dt_sub)
                if zen_sub >= np.pi / 2.0:
                    continue
                sun_vec = ul.vec_from_zen_azm(zen_sub, azm_sub)
                si = sky_disc.find_simplex(sun_vec[0], sun_vec[1], sun_vec[2])
                if si != -1:
                    sun_simplices.add(si)

        # Draw sun-path simplices with bold yellow edges
        for si in sun_simplices:
            node_inds = sky_disc.simplices[si]
            tri_azm = np.append(node_azm[node_inds], node_azm[node_inds[0]])
            tri_zen = np.append(node_zen_deg[node_inds], node_zen_deg[node_inds[0]])
            ax.plot(tri_azm, tri_zen, '-', color='gold', linewidth=1.5)

    ax.set_ylim(0.0, 90.0)
    ax.set_rlabel_position(135)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['S', 'W', 'N', 'E'])
    ax.set_title("Sky Dome Obstruction Map\n(red = blocked, blue = free)", pad=20)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

def plot_daily_profiles(results, save_path=None):
    """Plot average daily irradiance profiles for each month (12 subplots).

    Shows total obstructed/unobstructed and separate beam/diffuse components
    as hourly averages for each month. Subplot titles include daily cumulative
    radiation (with and without shading).

    Colors: blue = total, red = beam, green = diffuse.
    Solid = with shading, dashed = without shading (potential).

    Args:
        results: dict returned by compute_hourly_irradiance
        save_path: If provided, save figure to this path.
    """
    timestamps = results['timestamps']
    df = pd.DataFrame({
        'total_unobstructed': results['total_unobstructed'],
        'total_obstructed': results['total_obstructed'],
        'beam_unobstructed': results['beam_unobstructed'],
        'beam_obstructed': results['beam_obstructed'],
        'diffuse_unobstructed': results['diffuse_unobstructed'],
        'diffuse_obstructed': results['diffuse_obstructed'],
    }, index=timestamps)
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for month_idx in range(12):
        ax = axes[month_idx]
        month = month_idx + 1
        df_month = df[df['month'] == month]
        profile = df_month.groupby('hour').mean(numeric_only=True)

        # Average daily cumulative radiation [kWh/m²]
        daily_sums = df_month.groupby('date')[['total_unobstructed', 'total_obstructed']].sum()
        avg_daily_unobstructed = daily_sums['total_unobstructed'].mean() / 1000.0
        avg_daily_obstructed = daily_sums['total_obstructed'].mean() / 1000.0

        ax.plot(profile.index, profile['total_unobstructed'], 'b--', linewidth=1.5, label='Total (no shading)')
        ax.plot(profile.index, profile['total_obstructed'], 'b-', linewidth=1.5, label='Total (with shading)')
        ax.plot(profile.index, profile['beam_unobstructed'], 'r--', linewidth=1.0, label='Beam (no shading)')
        ax.plot(profile.index, profile['beam_obstructed'], 'r-', linewidth=1.0, label='Beam (with shading)')
        ax.plot(profile.index, profile['diffuse_unobstructed'], 'g--', linewidth=1.0, label='Diffuse (no shading)')
        ax.plot(profile.index, profile['diffuse_obstructed'], 'g-', linewidth=1.0, label='Diffuse (with shading)')

        month_name = pd.Timestamp(year=2000, month=month, day=1).strftime('%B')
        ax.set_title(f"{month_name}\n{avg_daily_obstructed:.2f} / {avg_daily_unobstructed:.2f} kWh/m²/d", fontsize=9)
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18])

    axes[0].legend(fontsize=7)
    fig.supxlabel('Hour of day')
    fig.supylabel('Irradiance [W/m²]')
    fig.suptitle('Average Daily Irradiance Profiles by Month (with shading / no shading)')
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

def plot_carpet(results, save_path=None):
    """Plot hourly carpet plots (x: day of year, y: hour of day, color: irradiance).

    Two subplots side by side with a dedicated colorbar axis.

    Args:
        results: dict returned by compute_hourly_irradiance
        save_path: If provided, save figure to this path.
    """
    timestamps = results['timestamps']
    n_hours = len(timestamps)
    n_days = n_hours // 24

    def reshape_to_carpet(data):
        """Reshape (n_hours,) to (24, n_days) with hour on y-axis, day on x-axis."""
        return data[:n_days * 24].reshape(n_days, 24).T

    carpet_unobstructed = reshape_to_carpet(results['total_unobstructed'])
    carpet_obstructed = reshape_to_carpet(results['total_obstructed'])

    vmax = max(np.max(carpet_unobstructed), np.max(carpet_obstructed))

    fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 0.03]})

    im1 = ax1.imshow(carpet_unobstructed, aspect='auto', origin='lower', vmin=0, vmax=vmax, cmap='hot', extent=[1, n_days, 0, 24])
    ax1.set_title('Unobstructed')
    ax1.set_xlabel('Day of year')
    ax1.set_ylabel('Hour of day')
    ax1.set_yticks([0, 6, 12, 18, 24])

    im2 = ax2.imshow(carpet_obstructed, aspect='auto', origin='lower', vmin=0, vmax=vmax, cmap='hot', extent=[1, n_days, 0, 24])
    ax2.set_title('Obstructed')
    ax2.set_xlabel('Day of year')
    ax2.set_yticks([0, 6, 12, 18, 24])

    fig.colorbar(im2, cax=cax, label='Irradiance [W/m²]')
    fig.suptitle('Hourly Irradiance Carpet Plots')
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    tmy3_filepath = "Freiburg-hour.csv"
    # pointcloud_filepath = "2026-06-06_20-20-43_wolfgang.csv"
    # pointcloud_filepath = "2026-06-06_20-24-20_wolfgang.csv"
    pointcloud_filepath = "2026-06-01_22-25-29.csv"

    # PV panel orientation
    panel_tilt_deg = 15.0        # 0 = horizontal, positive in direction of 'panel_azimuth_deg'
    panel_azimuth_deg = 0.  # 0 = facing South (Duffie-Beckman)
    print(f"panel_tilt_deg: {panel_tilt_deg:.1f}, panel_azimuth_deg: {panel_azimuth_deg:.1f}")

    # LiDAR alignment: clockwise angle from LiDAR y-axis to geographic South [deg]
    lidar_yaw_deg = -90.0 + 8.1

    # Offset from LiDAR position to evaluation point [m] in geographic frame
    # (x: West, y: South, z: up) — applied after rotation
    evaluation_offset = np.array([0.0, 0.0, 0.0])

    # Sky discretization resolution
    n_sky_nodes = 500
    max_points = int(1e6)
    min_points_per_patch = 10

    # PV efficiency
    panel_efficiency_stc = 0.20
    performance_ratio = 0.80

    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------

    # Load meteo data
    meteo_handler = MeteoDataHandlerTMY3(weather_file_path=tmy3_filepath)

    # Load and filter point cloud
    points_raw = load_point_cloud(pointcloud_filepath, max_points=max_points)

    # Transform point cloud to geographic / Duffie-Beckman frame
    rot_mat = Rotation.from_euler('z', -lidar_yaw_deg, degrees=True).as_matrix()
    points_geo = (points_raw @ rot_mat.T) - evaluation_offset

    # Re-filter after offset (some points may now be below evaluation plane)
    points_geo = filter_upper_hemisphere(points_geo)

    # Convert to spherical coordinates
    point_zen, point_azm = points_to_spherical(points_geo)

    # Build sky discretization and determine obstruction
    sky_disc = build_sky_discretization(n_nodes=n_sky_nodes)
    obstruction_mask = compute_obstruction_mask(sky_disc, point_zen, point_azm,
        min_points_per_patch=min_points_per_patch)

    # Compute panel normal
    panel_normal = compute_panel_normal(panel_tilt_deg, panel_azimuth_deg)

    # Compute hourly irradiance
    results = compute_hourly_irradiance(meteo_handler, panel_normal, obstruction_mask, sky_disc)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    # Compute annual sums and shading loss
    annual_beam_no_shading = np.sum(results['beam_unobstructed'])
    annual_beam_with_shading = np.sum(results['beam_obstructed'])
    annual_diffuse_no_shading = np.sum(results['diffuse_unobstructed'])
    annual_diffuse_with_shading = np.sum(results['diffuse_obstructed'])
    annual_with_shading = np.sum(results['total_obstructed'])
    annual_without_shading = np.sum(results['total_unobstructed'])
    collectable_fraction = annual_with_shading / annual_without_shading

    # PV system estimate
    annual_yield_with_shading = annual_with_shading * panel_efficiency_stc * performance_ratio
    annual_yield_no_shading = annual_without_shading * panel_efficiency_stc * performance_ratio

    print(f"Sky view factor: {results['sky_view_factor']:.4f}")
    print(f"Annual beam irradiation (no shading):   {annual_beam_no_shading / 1000:.1f} kWh/m²")
    print(f"Annual beam irradiation (with shading): {annual_beam_with_shading / 1000:.1f} kWh/m²")
    print(f"Annual diffuse irradiation (no shading):   {annual_diffuse_no_shading / 1000:.1f} kWh/m²")
    print(f"Annual diffuse irradiation (with shading): {annual_diffuse_with_shading / 1000:.1f} kWh/m²")
    print(f"Annual total irradiation (no shading):   {annual_without_shading / 1000:.1f} kWh/m²")
    print(f"Annual total irradiation (with shading): {annual_with_shading / 1000:.1f} kWh/m²")
    print(f"Collectable fraction: {collectable_fraction:.4f}")
    print(f"Annual shading loss: {(1 - collectable_fraction) * 100:.2f} %")
    print(f"Blocked patches: {np.sum(obstruction_mask)} / {len(obstruction_mask)}")
    print(f"")
    print(f"--- PV System Estimate (η_STC={panel_efficiency_stc:.0%}, PR={performance_ratio:.0%}) ---")
    print(f"Estimated annual yield (no shading):   {annual_yield_no_shading / 1000:.0f} kWh/m²/a")
    print(f"Estimated annual yield (with shading): {annual_yield_with_shading / 1000:.0f} kWh/m²/a")
    print(f"")
    print(f"--- Average Daily PV Yield by Month [kWh/m²/d] ---")
    print(f"{'Month':<12} {'No shading':>12} {'With shading':>14}")
    df_monthly = pd.DataFrame({
        'total_obstructed': results['total_obstructed'],
        'total_unobstructed': results['total_unobstructed'],
    }, index=results['timestamps'])
    df_monthly['date'] = df_monthly.index.date
    df_monthly['month'] = df_monthly.index.month
    for month in range(1, 13):
        df_month = df_monthly[df_monthly['month'] == month]
        daily_sums = df_month.groupby('date')[['total_unobstructed', 'total_obstructed']].sum()
        avg_no_shading = daily_sums['total_unobstructed'].mean() * panel_efficiency_stc * performance_ratio / 1000.0
        avg_with_shading = daily_sums['total_obstructed'].mean() * panel_efficiency_stc * performance_ratio / 1000.0
        month_name = pd.Timestamp(year=2000, month=month, day=1).strftime('%B')
        print(f"{month_name:<12} {avg_no_shading:>12.2f} {avg_with_shading:>14.2f}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    plot_sky_dome(sky_disc, obstruction_mask, meteo_handler=meteo_handler, save_path="sky_dome.png")
    plot_daily_profiles(results, save_path="daily_profiles.png")
    # plot_carpet(results, save_path="carpet_plot.pdf")
    plt.show(block=True)
