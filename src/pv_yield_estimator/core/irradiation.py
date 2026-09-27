"""Irradiation per sky patch: hourly direct radiation vectors per patch and isotropic diffuse radiation.

Independent of the obstruction and the panel orientation, so both can be varied without recomputing it.
"""

import logging

import numpy as np
import pandas as pd
import pvlib

from pv_yield_estimator.core.sky import SkyDiscretization, directions_from_zenith_azimuth
from pv_yield_estimator.file_format import FORMAT_VERSION_KEY, check_format_version

log = logging.getLogger(__name__)

PATCH_IRRADIATION_FORMAT_VERSION = 1
### Largest angular speed of the sun on the sky: 360° per day.
SUN_ANGULAR_SPEED_DEG_PER_HOUR = 15.0


class PatchIrradiation:
    """Irradiation per sky patch and hour.

    Direct radiation is stored sparsely, as entries (hour, patch, vector V = Σ E_k s_k) over the sub-steps k of the
    hour whose sun direction s_k lies in the patch, with E_k the direct normal irradiation of the sub-step in Wh/m².
    A panel with normal n receives max(V · n, 0) from an entry; |V| is the direct normal irradiation from the patch.
    Diffuse radiation is isotropic: the hourly diffuse horizontal irradiation (dhi, Wh/m²) together with the patch
    direction integrals of the sky discretization gives the diffuse radiation per patch for any orientation.
    ghi and dni are kept for reference. Hours are contiguous, starting at `time_start` (timezone-aware).
    """

    def __init__(self, sky, time_start, ghi, dhi, dni, direct_hours, direct_patches, direct_vectors, latitude, longitude, altitude=0.0, metadata=None):
        self.sky = sky
        self.time_start = pd.Timestamp(time_start)
        if self.time_start.tz is None:
            raise ValueError(f"Irradiation per sky patch needs a timezone-aware start time, got {time_start!r}")
        self.ghi, self.dhi, self.dni = (np.asarray(values, dtype=float) for values in (ghi, dhi, dni))
        self.direct_hours = np.asarray(direct_hours, dtype=int)
        self.direct_patches = np.asarray(direct_patches, dtype=int)
        self.direct_vectors = np.asarray(direct_vectors, dtype=float).reshape(-1, 3)
        self.latitude, self.longitude, self.altitude = latitude, longitude, altitude
        self.metadata = dict(metadata or {})

    @property
    def n_hours(self):
        """Number of hours."""
        return len(self.dhi)

    @property
    def index(self):
        """Start of each hour, timezone-aware."""
        return pd.date_range(self.time_start, periods=self.n_hours, freq="h")

    def annual_direct_vectors(self):
        """Sum of the direct radiation vectors per patch over all hours, shape (n_patches, 3), in Wh/m².

        Enough for the annual direct radiation on any panel orientation (orientation comparison).
        """
        return np.stack([np.bincount(self.direct_patches, weights=self.direct_vectors[:, i], minlength=self.sky.n_patches) for i in range(3)], axis=1)

    def annual_direct_normal_per_patch(self):
        """Annual direct normal irradiation per patch in Wh/m² (sum of |V|), independent of orientation, e.g. for the sky plot."""
        return np.bincount(self.direct_patches, weights=np.linalg.norm(self.direct_vectors, axis=1), minlength=self.sky.n_patches)

    def to_dict(self, decimals=4):
        """Content of the JSON file, with format version; values rounded to `decimals` (Wh/m²)."""
        return {
            FORMAT_VERSION_KEY: PATCH_IRRADIATION_FORMAT_VERSION, "kind": "patch_irradiation", "metadata": self.metadata,
            "latitude": self.latitude, "longitude": self.longitude, "altitude": self.altitude,
            "time_start": self.time_start.isoformat(), "n_hours": self.n_hours,
            **self.sky.to_dict(),
            "ghi": np.round(self.ghi, decimals).tolist(), "dhi": np.round(self.dhi, decimals).tolist(), "dni": np.round(self.dni, decimals).tolist(),
            "direct_hours": self.direct_hours.tolist(), "direct_patches": self.direct_patches.tolist(), "direct_vectors": np.round(self.direct_vectors, decimals).tolist(),
        }

    @classmethod
    def from_dict(cls, data, source="<dict>"):
        """Inverse of `to_dict`; checks the format version."""
        check_format_version(data, PATCH_IRRADIATION_FORMAT_VERSION, "Irradiation per sky patch", source)
        if data.get("kind") != "patch_irradiation":
            raise ValueError(f"{source} is not an irradiation per sky patch file (kind {data.get('kind')!r})")
        if len(data["dhi"]) != data["n_hours"]:
            raise ValueError(f"{source}: n_hours is {data['n_hours']}, but dhi has {len(data['dhi'])} values")
        return cls(SkyDiscretization.from_dict(data), data["time_start"], data["ghi"], data["dhi"], data["dni"], data["direct_hours"], data["direct_patches"], data["direct_vectors"], data["latitude"], data["longitude"], data.get("altitude", 0.0), data.get("metadata"))


def sub_steps_per_hour_for(sky, sub_steps_per_hour=None, max_sun_step_patch_fraction=0.5, **kwargs):
    """Number of sun position sub-steps per hour: given explicitly, or so that the sun moves at most the given fraction of a patch (mean edge) per step."""
    if sub_steps_per_hour is not None:
        return int(sub_steps_per_hour)
    return int(np.ceil(SUN_ANGULAR_SPEED_DEG_PER_HOUR / (max_sun_step_patch_fraction * np.degrees(sky.mean_edge_angle_rad()))))


def compute_patch_irradiation(weather, sky, latitude, longitude, altitude=0.0, sub_steps_per_hour=None, max_sun_step_patch_fraction=0.5, **kwargs):
    """Irradiation per sky patch from hourly weather data (see `PatchIrradiation`).

    Each hour is split into equal sub-steps with the sun position (pvlib, apparent position including refraction) at
    their midpoints. The hour's direct normal irradiation is distributed evenly over the sub-steps with the sun above
    the horizon, so it is conserved also in sunrise and sunset hours.
    """
    n_sub_steps = sub_steps_per_hour_for(sky, sub_steps_per_hour, max_sun_step_patch_fraction)
    hourly = weather.hourly
    n_hours = len(hourly)
    offsets = pd.to_timedelta((np.arange(n_sub_steps) + 0.5) / n_sub_steps, unit="h")
    times = pd.DatetimeIndex((hourly.index.values[:, None] + offsets.values[None, :]).ravel()).tz_localize("UTC").tz_convert(hourly.index.tz)
    solar_position = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude)
    directions = directions_from_zenith_azimuth(np.radians(solar_position["apparent_zenith"].to_numpy()), np.radians(solar_position["azimuth"].to_numpy())).reshape(n_hours, n_sub_steps, 3)
    sun_up = directions[:, :, 2] > 0.0
    n_sun_up = sun_up.sum(axis=1)
    dni = hourly["dni"].to_numpy(dtype=float)
    lost = (dni > 0) & (n_sun_up == 0)
    if np.any(lost):
        log.warning(f"{np.count_nonzero(lost)} hours with direct radiation but the sun below the horizon at all sub-steps; dropping {dni[lost].sum() / 1000:.3f} kWh/m² DNI")
    ### Direct normal irradiation per sub-step (Wh/m²), for sub-steps with the sun up in hours with direct radiation.
    energy_per_sub_step = np.where(sun_up & (dni > 0)[:, None], dni[:, None] / np.maximum(n_sun_up, 1)[:, None], 0.0)
    hour_indices, sub_step_indices = np.nonzero(energy_per_sub_step)
    sun_directions = directions[hour_indices, sub_step_indices]
    patches = sky.find_patches(sun_directions)
    vectors = energy_per_sub_step[hour_indices, sub_step_indices, None] * sun_directions
    ### Combine the sub-steps of an hour that fall into the same patch.
    keys, entry_of_sub_step = np.unique(hour_indices * sky.n_patches + patches, return_inverse=True)
    direct_vectors = np.stack([np.bincount(entry_of_sub_step, weights=vectors[:, i], minlength=len(keys)) for i in range(3)], axis=1)
    irradiation = PatchIrradiation(sky, hourly.index[0], hourly["ghi"], hourly["dhi"], dni, keys // sky.n_patches, keys % sky.n_patches, direct_vectors, latitude, longitude, altitude, metadata={"weather_source": weather.source, "weather_name": weather.name, "sub_steps_per_hour": n_sub_steps})
    log.info(f"Irradiation per sky patch: {n_hours} hours, {n_sub_steps} sub-steps per hour, {len(keys)} direct entries on {len(np.unique(irradiation.direct_patches))} patches, annual DNI {np.linalg.norm(direct_vectors, axis=1).sum() / 1000:.1f} kWh/m²")
    return irradiation
