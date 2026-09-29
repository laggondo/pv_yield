"""Interactive Bokeh plots, shared by the front ends: the CLI saves them as standalone HTML, the browser embeds them as JSON (`json_item`).

Units: radiation per patch and year in kWh/m², hourly radiation on the panel in Wh/m² (equal to the mean irradiance in W/m²).
"""

import numpy as np
import pandas as pd
from bokeh.layouts import column, gridplot
from bokeh.models import ColorBar, ColumnDataSource, FixedTicker, HoverTool, Legend, LinearColorMapper
from bokeh.palettes import Inferno256
from bokeh.plotting import figure

from pv_yield_estimator.core.irradiation import sun_path_directions
from pv_yield_estimator.core.panel_yield import SHADING_CASES
from pv_yield_estimator.core.sky import zenith_azimuth_from_directions

### Days of the sun paths drawn into the sky plot: summer solstice, equinox, winter solstice.
SUN_PATH_DAYS = (("06-21", "21 June", "#2ca02c"), ("03-20", "20 March / 23 September", "#17becf"), ("12-21", "21 December", "#9467bd"))
CASE_STYLES = {"unobstructed": {"line_color": "#7f7f7f", "line_dash": "dashed"}, "obstructed": {"line_color": "#d95f02", "line_dash": "solid"}}
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
OBSTRUCTED_HATCH_COLOR = "#1f77b4"
EMPTY_PATCH_COLOR = "#f4f4f4"
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


def polar_plot_xy(directions):
    """Equidistant polar projection of directions (x east, y north, z up): radius = zenith angle in degrees, north up, east right."""
    zenith, azimuth = zenith_azimuth_from_directions(directions)
    zenith_deg = np.degrees(zenith)
    return zenith_deg * np.sin(azimuth), zenith_deg * np.cos(azimuth)


def annual_radiation_per_patch(irradiation):
    """Annual direct, diffuse and total radiation per patch in kWh/m², independent of the panel orientation.

    Direct: direct normal irradiation from the patch (sum of |V|). Diffuse: isotropic sky radiance DHI / π times the
    patch's solid angle, i.e. what a surface facing the patch receives from it.
    """
    direct = irradiation.annual_direct_normal_per_patch() / 1000.0
    diffuse = irradiation.dhi.sum() / np.pi * irradiation.sky.solid_angles() / 1000.0
    return {"direct": direct, "diffuse": diffuse, "total": direct + diffuse}


def sky_hemisphere_plot(obstructed_sky, irradiation=None, width=640, height=560):
    """Sky hemisphere as a map (north up, east right, radius = zenith angle): patches coloured by annual radiation, obstructed patches hatched, sun paths.

    Without `irradiation`, only the patches and obstructions are drawn. Shows how much radiation each blocked part
    of the sky costs, independent of the panel orientation.
    """
    sky = obstructed_sky.sky
    patch_x, patch_y = polar_plot_xy(sky.nodes[sky.triangles])
    zenith, azimuth = zenith_azimuth_from_directions(sky.patch_centers())
    data = {"xs": list(patch_x), "ys": list(patch_y), "elevation_deg": 90.0 - np.degrees(zenith), "azimuth_deg": np.degrees(azimuth),
            "obstructed": np.where(obstructed_sky.obstructed, "yes", "no"), "hatch": np.where(obstructed_sky.obstructed, "x", " ")}
    tooltips = [("elevation", "@elevation_deg{0.0}°"), ("azimuth", "@azimuth_deg{0.0}°"), ("obstructed", "@obstructed")]
    title = "Sky hemisphere: obstructed patches hatched"
    if irradiation is not None:
        radiation = annual_radiation_per_patch(irradiation)
        data |= {f"{component}_kwh_m2": values for component, values in radiation.items()}
        tooltips += [("direct normal", "@direct_kwh_m2{0.00} kWh/m²"), ("diffuse", "@diffuse_kwh_m2{0.00} kWh/m²"), ("total", "@total_kwh_m2{0.00} kWh/m²")]
        title = "Sky hemisphere: annual radiation per patch (kWh/m²), obstructed patches hatched"
    source = ColumnDataSource(data)
    plot = figure(title=title, x_range=(-100, 100), y_range=(-100, 100), width=width, height=height, sizing_mode="scale_width", match_aspect=True,
                  tools=TOOLS + ",tap", active_scroll="wheel_zoom", x_axis_label="west ← → east (zenith angle, °)", y_axis_label="south ↓ ↑ north (zenith angle, °)")
    if irradiation is not None:
        color_mapper = LinearColorMapper(palette=Inferno256, low=0.0, high=float(radiation["total"].max()))
        fill = {"field": "total_kwh_m2", "transform": color_mapper}
        plot.add_layout(ColorBar(color_mapper=color_mapper, title="kWh/m²"), "right")
    else:
        fill = EMPTY_PATCH_COLOR
    patches = plot.patches("xs", "ys", source=source, fill_color=fill, line_color="#999999", line_width=0.3, hatch_pattern="hatch", hatch_color=OBSTRUCTED_HATCH_COLOR, hatch_alpha=0.8)
    plot.add_tools(HoverTool(renderers=[patches], tooltips=tooltips))
    if irradiation is not None:
        legend_items = []
        year = irradiation.time_start.year
        for month_day, label, color in SUN_PATH_DAYS:
            x, y = polar_plot_xy(sun_path_directions(f"{year}-{month_day}", irradiation.latitude, irradiation.longitude, irradiation.altitude, irradiation.time_start.tz))
            legend_items.append((f"sun path {label}", [plot.line(x, y, line_color=color, line_width=2)]))
        plot.add_layout(Legend(items=legend_items, location="bottom_right", background_fill_alpha=0.7))
    angle = np.linspace(0, 2 * np.pi, 361)
    plot.line(90 * np.sin(angle), 90 * np.cos(angle), line_color="black")
    plot.text([0, 95, 0, -95], [95, 0, -95, 0], text=["N", "E", "S", "W"], text_align="center", text_baseline="middle")
    return plot


def hour_axis_offset(index):
    """Offset in hours of the hour starts from full hours (e.g. 0.68 for PVGIS data), identical for all hours of contiguous hourly data."""
    first = index[0]
    return first.minute / 60.0 + first.second / 3600.0


def carpet_plots(result, component="total", width=900, height=260):
    """Carpet plots of the hourly radiation on the panel over the year (day of year vs. hour of day), unobstructed and obstructed, with one shared colour scale."""
    hourly = result.hourly
    offset = hour_axis_offset(hourly.index)
    tables = {case: hourly.pivot_table(index=hourly.index.hour, columns=hourly.index.dayofyear, values=f"{component}_{case}", aggfunc="sum").reindex(range(24)) for case in SHADING_CASES}
    first_day, last_day = tables["unobstructed"].columns.min(), tables["unobstructed"].columns.max()
    color_mapper = LinearColorMapper(palette=Inferno256, low=0.0, high=float(max(table.max().max() for table in tables.values())), nan_color="white")
    month_starts = pd.date_range(f"{hourly.index[0].year}-01-01", periods=12, freq="MS")
    plots = []
    for case in SHADING_CASES:
        plot = figure(title=f"Hourly {component} radiation on the panel (Wh/m²), {case}", x_range=plots[0].x_range if plots else (first_day - 0.5, last_day + 0.5), y_range=(0, 24),
                      width=width, height=height, sizing_mode="scale_width", tools=TOOLS, active_scroll="wheel_zoom", x_axis_label="day of year", y_axis_label="hour of day (local standard time)")
        image = plot.image(image=[tables[case].to_numpy()], x=first_day - 0.5, y=offset, dw=last_day - first_day + 1, dh=24, color_mapper=color_mapper)
        plot.add_tools(HoverTool(renderers=[image], tooltips=[("day of year", "$x{0}"), ("hour of day", "$y{0.0}"), ("radiation", "@image{0} Wh/m²")]))
        plot.xaxis.ticker = FixedTicker(ticks=list(month_starts.dayofyear))
        plot.xaxis.major_label_overrides = {day: name for day, name in zip(month_starts.dayofyear, MONTH_NAMES)}
        plot.yaxis.ticker = FixedTicker(ticks=list(range(0, 25, 3)))
        plot.add_layout(ColorBar(color_mapper=color_mapper, title="Wh/m²"), "right")
        plots.append(plot)
    return column(plots, sizing_mode="scale_width")


def monthly_profiles_plot(result, component="total", n_columns=4, width=220, height=180):
    """Average daily profile of the radiation on the panel per month (W/m²), unobstructed and obstructed, one small plot per month."""
    profiles = result.monthly_daily_profiles()
    offset = hour_axis_offset(result.hourly.index) + 0.5   ### centre of each hour
    y_max = 1.05 * float(profiles[[f"{component}_{case}" for case in SHADING_CASES]].max().max())
    plots = []
    for month in profiles.index.get_level_values("month").unique():
        profile = profiles.loc[month]
        plot = figure(title=MONTH_NAMES[month - 1], x_range=plots[0].x_range if plots else (0, 24), y_range=plots[0].y_range if plots else (0, y_max), width=width, height=height, tools=TOOLS)
        for case in SHADING_CASES:
            source = ColumnDataSource({"hour": profile.index.to_numpy() + offset, "radiation": profile[f"{component}_{case}"].to_numpy()})
            line = plot.line("hour", "radiation", source=source, line_width=2, legend_label=case, **CASE_STYLES[case])
            plot.add_tools(HoverTool(renderers=[line], tooltips=[("case", case), ("hour", "@hour{0.0}"), ("radiation", "@radiation{0} W/m²")], mode="vline"))
        plot.xaxis.ticker = FixedTicker(ticks=list(range(0, 25, 6)))
        plot.legend.visible = not plots
        plot.legend.location = "top_left"
        plot.legend.label_text_font_size = "8pt"
        plots.append(plot)
    return gridplot(plots, ncols=n_columns, sizing_mode="scale_width", merge_tools=True)


def result_plots(result, obstructed_sky, irradiation):
    """All plots of a yield result by name, in display order; each value is a Bokeh model for `json_item` or `save`."""
    return {"sky_hemisphere": sky_hemisphere_plot(obstructed_sky, irradiation), "monthly_profiles": monthly_profiles_plot(result), "carpet": carpet_plots(result)}
