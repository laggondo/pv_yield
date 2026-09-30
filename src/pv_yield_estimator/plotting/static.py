"""Static matplotlib versions of the plots, for PNG files and the PDF report; the same content as the interactive Bokeh plots.

Figures are created with `matplotlib.figure.Figure` directly (no pyplot), so no GUI backend or global state is
involved; this also works in Pyodide. Units as in `plotting.interactive`.
"""

import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from pv_yield_estimator.core.irradiation import sun_path_directions
from pv_yield_estimator.core.panel_yield import SHADING_CASES
from pv_yield_estimator.plotting.interactive import (BEST_MARKER_COLOR, CASE_BAR_COLORS, CASE_DASHES, COMPONENT_COLORS, EMPTY_PATCH_COLOR, MONTH_NAMES, OBSTRUCTED_HATCH_COLOR, PANEL_MARKER_COLOR,
                                                     SUN_PATH_DAYS, annual_radiation_per_patch, hour_axis_offset, orientation_contours, polar_plot_xy)

COLORMAP = "inferno"
### A4 landscape in inches, the page size of the PDF report.
PAGE_SIZE = (11.69, 8.27)


def sky_hemisphere_figure(obstructed_sky, irradiation=None, figsize=(7.0, 7.5)):
    """Sky hemisphere (north up, east right, radius = zenith angle): patches coloured by annual radiation, obstructed patches hatched, sun paths."""
    figure = Figure(figsize=figsize, layout="constrained")
    axes = figure.add_subplot()
    sky = obstructed_sky.sky
    patch_x, patch_y = polar_plot_xy(sky.nodes[sky.triangles])
    polygons = np.stack([patch_x, patch_y], axis=-1)
    if irradiation is not None:
        radiation = annual_radiation_per_patch(irradiation)["total"]
        patches = PolyCollection(polygons, array=radiation, cmap=COLORMAP, norm=Normalize(0.0, radiation.max()), edgecolors="#999999", linewidths=0.2)
        figure.colorbar(patches, ax=axes, orientation="horizontal", shrink=0.8, pad=0.02, label="annual radiation per patch (kWh/m²)")
        axes.set_title("Annual radiation per sky patch; hatched: obstructed")
    else:
        patches = PolyCollection(polygons, facecolors=EMPTY_PATCH_COLOR, edgecolors="#999999", linewidths=0.2)
        axes.set_title("Sky hemisphere: obstructed patches hatched")
    axes.add_collection(patches)
    axes.add_collection(PolyCollection(polygons[obstructed_sky.obstructed], facecolors="none", edgecolors=OBSTRUCTED_HATCH_COLOR, linewidths=0.0, hatch="xxx"))
    if irradiation is not None:
        year = irradiation.time_start.year
        for month_day, label, color in SUN_PATH_DAYS:
            x, y = polar_plot_xy(sun_path_directions(f"{year}-{month_day}", irradiation.latitude, irradiation.longitude, irradiation.altitude, irradiation.time_start.tz))
            axes.plot(x, y, color=color, linewidth=2, label=f"sun path {label}")
        axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=3, fontsize=7, frameon=False)
    angle = np.linspace(0, 2 * np.pi, 361)
    axes.plot(90 * np.sin(angle), 90 * np.cos(angle), color="black", linewidth=1)
    for elevation_deg in (30, 60):
        axes.plot((90 - elevation_deg) * np.sin(angle), (90 - elevation_deg) * np.cos(angle), color="white", alpha=0.6, linestyle="dotted")
        axes.text(0, elevation_deg - 90, f"{elevation_deg}°", color="white", fontsize=8, ha="center", va="bottom")
    for x, y, text in ((0, 96, "N"), (96, 0, "E"), (0, -96, "S"), (-96, 0, "W")):
        axes.text(x, y, text, ha="center", va="center")
    axes.set_xlim(-100, 100)
    axes.set_ylim(-100, 100)
    axes.set_aspect("equal")
    axes.axis("off")
    return figure


def monthly_profiles_figure(result, figsize=PAGE_SIZE):
    """Average daily profile per month (W/m²): total, direct and diffuse, each unobstructed (dashed) and obstructed (solid), in a 3 × 4 grid."""
    profiles = result.monthly_daily_profiles()
    offset = hour_axis_offset(result.hourly.index) + 0.5
    figure = Figure(figsize=figsize, layout="constrained")
    axes_grid = figure.subplots(3, 4, sharex=True, sharey=True)
    for axes, month in zip(axes_grid.flat, profiles.index.get_level_values("month").unique()):
        profile = profiles.loc[month]
        for component, color in COMPONENT_COLORS.items():
            for case, dash in CASE_DASHES.items():
                axes.plot(profile.index.to_numpy() + offset, profile[f"{component}_{case}"], color=color, linestyle=dash, linewidth=1.8 if component == "total" else 1.2,
                          label=f"{component}, {case}")
        axes.set_title(MONTH_NAMES[month - 1], fontsize=9)
        axes.set_xlim(0, 24)
        axes.set_xticks(range(0, 25, 6))
        axes.grid(alpha=0.3)
    axes_grid[0, 0].set_ylim(bottom=0)
    for axes in axes_grid[:, 0]:
        axes.set_ylabel("W/m²")
    for axes in axes_grid[-1, :]:
        axes.set_xlabel("hour of day")
    figure.legend(*axes_grid[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=6, fontsize=8, frameon=False)
    figure.suptitle("Average daily profile of the radiation on the panel per month")
    return figure


def daily_bars_figure(result, figsize=(11.0, 4.0)):
    """Radiation on the panel per day (kWh/m²): unobstructed (light) behind obstructed (dark)."""
    daily = result.daily()
    days = daily.index.tz_localize(None) + pd.Timedelta(hours=12)
    figure = Figure(figsize=figsize, layout="constrained")
    axes = figure.add_subplot()
    for case in SHADING_CASES:
        axes.bar(days, daily[f"total_{case}"], width=0.85, color=CASE_BAR_COLORS[case], label=case)
    axes.set_ylabel("kWh/m² per day")
    axes.set_title("Radiation on the panel per day")
    month_starts = pd.date_range(f"{days[0].year}-01-01", periods=12, freq="MS")
    axes.set_xticks(month_starts, MONTH_NAMES)
    axes.set_xlim(days[0] - pd.Timedelta(days=1), days[-1] + pd.Timedelta(days=1))
    axes.grid(axis="y", alpha=0.3)
    axes.legend(loc="upper left", ncols=2, frameon=False)
    return figure


def carpet_figure(result, component="total", figsize=PAGE_SIZE):
    """Carpet plots of the hourly radiation on the panel (hour of day across, day of year downwards), unobstructed and obstructed with one colour scale."""
    hourly = result.hourly
    offset = hour_axis_offset(hourly.index)
    tables = {case: hourly.pivot_table(index=hourly.index.dayofyear, columns=hourly.index.hour, values=f"{component}_{case}", aggfunc="sum").reindex(columns=range(24)) for case in SHADING_CASES}
    first_day, last_day = tables["unobstructed"].index.min(), tables["unobstructed"].index.max()
    norm = Normalize(0.0, max(float(table.max().max()) for table in tables.values()))
    month_starts = pd.date_range(f"{hourly.index[0].year}-01-01", periods=12, freq="MS")
    figure = Figure(figsize=figsize, layout="constrained")
    axes_pair = figure.subplots(1, 2, sharey=True)
    for axes, case in zip(axes_pair, SHADING_CASES):
        image = tables[case].to_numpy()
        ### The part of each day's last hour past 24:00 is drawn again at the start of the next day, as in the interactive plot.
        for shift, day_shift in ((0.0, 0.0), (-24.0, 1.0)):
            mappable = axes.imshow(image, extent=(offset + shift, offset + shift + 24, last_day + 0.5 + day_shift, first_day - 0.5 + day_shift), aspect="auto", cmap=COLORMAP, norm=norm, interpolation="nearest")
        axes.set_xlim(0, 24)
        axes.set_ylim(last_day + 0.5, first_day - 0.5)
        axes.set_xticks(range(0, 25, 3))
        axes.set_yticks(month_starts.dayofyear, MONTH_NAMES)
        axes.set_xlabel("hour of day (local standard time)")
        axes.set_title(case.capitalize())
    figure.colorbar(mappable, ax=axes_pair, orientation="horizontal", shrink=0.6, label=f"hourly {component} radiation on the panel (Wh/m²)")
    return figure


def orientation_figure(result, figsize=(11.0, 4.5)):
    """Annual radiation on the panel over azimuth and tilt (kWh/m²), unobstructed and obstructed with one colour scale; best and panel orientation marked, contours at 95 % and 90 % of the best."""
    grid = result.orientation_grid
    norm = Normalize(0.0, float(grid.radiation["unobstructed"].max()))
    figure = Figure(figsize=figsize, layout="constrained")
    axes_pair = figure.subplots(1, 2, sharey=True)
    for axes, case in zip(axes_pair, SHADING_CASES):
        best = grid.best(case)
        mappable = axes.pcolormesh(grid.azimuths_deg, grid.tilts_deg, grid.radiation[case], cmap=COLORMAP, norm=norm, shading="nearest")
        for fraction, x, y in orientation_contours(grid, case):
            axes.plot(x, y, color="white", linestyle="dotted", linewidth=1.2)
            axes.text(x[len(x) // 2], y[len(y) // 2], f"{fraction:.0%}", color="white", fontsize=7, ha="center", va="bottom")
        axes.plot(best["azimuth_deg"], best["tilt_deg"], marker="*", markersize=14, color=BEST_MARKER_COLOR, markeredgecolor="black", linestyle="none", label="best")
        axes.plot(result.azimuth_deg, result.tilt_deg, marker="o", markersize=8, color=PANEL_MARKER_COLOR, markeredgecolor="black", linestyle="none", label="panel")
        axes.set_xticks(range(0, 361, 45))
        axes.set_xlabel("panel azimuth (°, compass: 90 = east, 180 = south)")
        axes.set_title(f"{case.capitalize()}: best tilt {best['tilt_deg']:g}°, azimuth {best['azimuth_deg']:g}°, {best['radiation_kwh_m2']:.0f} kWh/m²", fontsize=10)
        axes.legend(loc="upper right", fontsize=8)
    axes_pair[0].set_ylabel("panel tilt (°)")
    figure.colorbar(mappable, ax=axes_pair, orientation="horizontal", shrink=0.6, label="annual radiation on the panel (kWh/m²)")
    return figure


def static_figures(result, obstructed_sky, irradiation):
    """All static figures of a yield result by name, in the order of the interactive plots."""
    return {"sky_hemisphere": sky_hemisphere_figure(obstructed_sky, irradiation), "monthly_profiles": monthly_profiles_figure(result), "daily_bars": daily_bars_figure(result),
            "carpet": carpet_figure(result), "orientation": orientation_figure(result)}
