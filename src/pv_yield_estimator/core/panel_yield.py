"""Radiation on the panel, PV yield and key figures, from the irradiation per sky patch and the obstructed sky."""

import logging

import numpy as np
import pandas as pd

from pv_yield_estimator.core.sky import panel_normal

log = logging.getLogger(__name__)

RADIATION_COMPONENTS = ("direct", "diffuse", "total")
SHADING_CASES = ("unobstructed", "obstructed")


def compute_panel_radiation(irradiation, obstructed_sky, tilt_deg=30.0, azimuth_deg=180.0, **kwargs):
    """Hourly radiation on the panel in Wh/m², obstructed (obstructed patches excluded) and unobstructed.

    Columns `<component>_<case>` for component direct, diffuse, total and case unobstructed, obstructed; index = hour
    starts. Direct: max(V · n, 0) per entry of the irradiation per sky patch. Diffuse: isotropic sky, each patch
    weighted by its direction integral · n / π, so a horizontal unobstructed panel receives exactly the diffuse
    horizontal irradiation and a tilted one (1 + cos tilt) / 2 of it. Ground-reflected radiation is not considered.
    """
    if not irradiation.sky.same_as(obstructed_sky.sky):
        raise ValueError(f"Irradiation per sky patch ({irradiation.sky.n_patches} patches) and obstructed sky description ({obstructed_sky.sky.n_patches} patches) use different sky discretizations; compute the irradiation with the sky of the obstructed sky description")
    normal = panel_normal(tilt_deg, azimuth_deg)
    free = ~obstructed_sky.obstructed
    direct_per_entry = np.maximum(irradiation.direct_vectors @ normal, 0.0)
    direct_unobstructed = np.bincount(irradiation.direct_hours, weights=direct_per_entry, minlength=irradiation.n_hours)
    direct_obstructed = np.bincount(irradiation.direct_hours, weights=direct_per_entry * free[irradiation.direct_patches], minlength=irradiation.n_hours)
    diffuse_weights = np.maximum(irradiation.sky.direction_integrals() @ normal, 0.0) / np.pi
    hourly = pd.DataFrame({
        "direct_unobstructed": direct_unobstructed, "direct_obstructed": direct_obstructed,
        "diffuse_unobstructed": irradiation.dhi * diffuse_weights.sum(), "diffuse_obstructed": irradiation.dhi * diffuse_weights[free].sum(),
    }, index=irradiation.index)
    for case in SHADING_CASES:
        hourly[f"total_{case}"] = hourly[f"direct_{case}"] + hourly[f"diffuse_{case}"]
    return hourly


class YieldResult:
    """Radiation on the panel and PV yield: hourly values, aggregates and key figures.

    Consumed by the plots, the export and the front ends. Radiation is in kWh/m² (hourly values in Wh/m²).
    PV yield = radiation × area × efficiency × performance ratio; specific yield (kWh/kWp) = radiation (kWh/m²) ×
    performance ratio, since 1 kWp corresponds to 1 kW/m² irradiance at standard test conditions.
    """

    def __init__(self, hourly, sky_view_factor, sky_view_factor_horizontal, tilt_deg=30.0, azimuth_deg=180.0, area_m2=1.0, efficiency=0.20, performance_ratio=0.80, **kwargs):
        self.hourly = hourly
        self.sky_view_factor = sky_view_factor
        self.sky_view_factor_horizontal = sky_view_factor_horizontal
        self.tilt_deg, self.azimuth_deg = tilt_deg, azimuth_deg
        self.area_m2, self.efficiency, self.performance_ratio = area_m2, efficiency, performance_ratio

    @property
    def yield_per_radiation(self):
        """PV yield in kWh per kWh/m² radiation on the panel: area × efficiency × performance ratio."""
        return self.area_m2 * self.efficiency * self.performance_ratio

    @property
    def rated_power_kwp(self):
        """Rated panel power in kWp: area × efficiency × 1 kW/m²."""
        return self.area_m2 * self.efficiency

    def daily(self):
        """Radiation per day in kWh/m², days in the time zone of the weather data."""
        return self.hourly.resample("D").sum() / 1000.0

    def monthly_daily_average(self):
        """Average daily radiation per month in kWh/m²/d (index: month 1–12), plus the yield columns in kWh/d."""
        monthly = self.daily().groupby(lambda day: day.month).mean()
        monthly.index.name = "month"
        for case in SHADING_CASES:
            monthly[f"yield_{case}"] = monthly[f"total_{case}"] * self.yield_per_radiation
        return monthly

    def monthly_daily_profiles(self):
        """Average daily profile per month in W/m² (mean over the month's days per hour of the day), index (month, hour)."""
        profiles = self.hourly.groupby([self.hourly.index.month, self.hourly.index.hour]).mean()
        profiles.index.names = ["month", "hour"]
        return profiles

    def key_figures(self):
        """Annual key figures as a flat dict of plain floats."""
        annual = self.hourly.sum() / 1000.0
        figures = {f"annual_{column}_kwh_m2": float(annual[column]) for column in self.hourly}
        for case in SHADING_CASES:
            figures[f"annual_yield_{case}_kwh"] = float(annual[f"total_{case}"] * self.yield_per_radiation)
            figures[f"specific_yield_{case}_kwh_kwp"] = float(annual[f"total_{case}"] * self.performance_ratio)
        figures["shading_loss"] = float(1.0 - annual["total_obstructed"] / annual["total_unobstructed"])
        figures["direct_shading_loss"] = float(1.0 - annual["direct_obstructed"] / annual["direct_unobstructed"])
        figures["sky_view_factor"] = self.sky_view_factor
        figures["sky_view_factor_horizontal"] = self.sky_view_factor_horizontal
        figures["rated_power_kwp"] = self.rated_power_kwp
        return figures


class YieldEstimator:
    """Combines the irradiation per sky patch, the obstructed sky description and the panel config into a `YieldResult`.

    Takes the whole config as `YieldEstimator(irradiation, obstructed_sky, **config)`; only the panel section is
    used, other sections are absorbed.
    """

    def __init__(self, irradiation, obstructed_sky, panel=None, **kwargs):
        self.irradiation = irradiation
        self.obstructed_sky = obstructed_sky
        self.panel = dict(panel or {})

    def run(self):
        """Compute the radiation on the panel and wrap it with the panel parameters into a result."""
        hourly = compute_panel_radiation(self.irradiation, self.obstructed_sky, **self.panel)
        result = YieldResult(hourly, self.panel_sky_view_factor(**self.panel), self.obstructed_sky.sky_view_factor(), **self.panel)
        log.info(f"Panel tilt {result.tilt_deg}°, azimuth {result.azimuth_deg}°: annual radiation {hourly['total_obstructed'].sum() / 1000:.1f} kWh/m² obstructed, {hourly['total_unobstructed'].sum() / 1000:.1f} unobstructed")
        return result

    def panel_sky_view_factor(self, tilt_deg=30.0, azimuth_deg=180.0, **kwargs):
        """Sky view factor of the tilted panel: fraction of isotropic diffuse sky radiation on it that passes the obstructions."""
        return self.obstructed_sky.sky_view_factor(panel_normal(tilt_deg, azimuth_deg))
