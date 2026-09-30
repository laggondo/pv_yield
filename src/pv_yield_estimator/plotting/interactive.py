"""Interactive Bokeh plots, shared by the front ends: the CLI saves them as standalone HTML, the browser embeds them as JSON (`json_item`).

Units: radiation per patch and year in kWh/m², hourly radiation on the panel in Wh/m² (equal to the mean irradiance in W/m²).
"""

import contourpy
import numpy as np
import pandas as pd
from bokeh.models import ColorBar, ColumnDataSource, DatetimeTickFormatter, FixedTicker, HoverTool, Legend, LegendItem, LinearColorMapper, Row
from bokeh.palettes import Inferno256
from bokeh.plotting import figure

from pv_yield_estimator.core.irradiation import sun_path_directions
from pv_yield_estimator.core.panel_yield import SHADING_CASES
from pv_yield_estimator.core.sky import zenith_azimuth_from_directions

### Days of the sun paths drawn into the sky plot: summer solstice, equinox, winter solstice.
SUN_PATH_DAYS = (("06-21", "21 June", "#2ca02c"), ("03-20", "20 March / 23 September", "#17becf"), ("12-21", "21 December", "#9467bd"))
CASE_DASHES = {"unobstructed": "dashed", "obstructed": "solid"}
### Daily bars: one hue, light for the unobstructed radiation (the potential) behind dark for the obstructed.
CASE_BAR_COLORS = {"unobstructed": "#f6c49a", "obstructed": "#d95f02"}
### Markers in the orientation heatmap, and its contour lines at these fractions of the best annual radiation.
BEST_MARKER_COLOR, PANEL_MARKER_COLOR = "#00e5ff", "#ffffff"
ORIENTATION_CONTOUR_FRACTIONS = (0.95, 0.9)
COMPONENT_COLORS = {"total": "#222222", "direct": "#d95f02", "diffuse": "#1f78b4"}
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
OBSTRUCTED_HATCH_COLOR = "#1f77b4"
EMPTY_PATCH_COLOR = "#f4f4f4"
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
### No pan or wheel zoom active by default, so touching or scrolling over a plot scrolls the page (phones); the tools are
### switched on in the toolbar.
INACTIVE_TOOLS = {"tools": TOOLS, "active_drag": None, "active_scroll": None}


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


def sky_hemisphere_plot(obstructed_sky, irradiation=None, width=600, height=780):
    """Sky hemisphere as a map (north up, east right, radius = zenith angle): patches coloured by annual radiation, obstructed patches hatched, sun paths.

    Without `irradiation`, only the patches and obstructions are drawn. Shows how much radiation each blocked part
    of the sky costs, independent of the panel orientation. Colour bar and legend sit below the hemisphere, so it
    stays round on narrow (phone) screens.
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
        title = "Annual radiation per sky patch; hatched: obstructed"
    source = ColumnDataSource(data)
    ### Automatic ranges (from the data: horizon circle and N/E/S/W labels) so that match_aspect keeps the hemisphere round.
    plot = figure(title=title, width=width, height=height, sizing_mode="scale_width", match_aspect=True,
                  **INACTIVE_TOOLS, x_axis_type=None, y_axis_type=None)
    plot.grid.visible = False
    if irradiation is not None:
        color_mapper = LinearColorMapper(palette=Inferno256, low=0.0, high=float(radiation["total"].max()))
        fill = {"field": "total_kwh_m2", "transform": color_mapper}
        plot.add_layout(ColorBar(color_mapper=color_mapper, title="annual radiation per patch (kWh/m²)", orientation="horizontal", height=12), "below")
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
        plot.add_layout(Legend(items=legend_items, orientation="vertical", border_line_alpha=0), "below")
    angle = np.linspace(0, 2 * np.pi, 361)
    plot.line(90 * np.sin(angle), 90 * np.cos(angle), line_color="black")
    ### Rings of constant elevation instead of axes (radius = zenith angle).
    for elevation_deg in (30, 60):
        plot.line((90 - elevation_deg) * np.sin(angle), (90 - elevation_deg) * np.cos(angle), line_color="white", line_alpha=0.6, line_dash="dotted")
        plot.text([0], [elevation_deg - 90], text=[f"{elevation_deg}°"], text_color="white", text_font_size="9pt", text_align="center", text_baseline="bottom")
    plot.text([0, 95, 0, -95], [95, 0, -95, 0], text=["N", "E", "S", "W"], text_align="center", text_baseline="middle")
    return plot


def hour_axis_offset(index):
    """Offset in hours of the hour starts from full hours (e.g. 0.68 for PVGIS data), identical for all hours of contiguous hourly data."""
    first = index[0]
    return first.minute / 60.0 + first.second / 3600.0


def carpet_plots(result, component="total", width=350, height=640):
    """Carpet plots of the hourly radiation on the panel over the year (hour of day across, day of year downwards), unobstructed and obstructed, with one shared colour scale.

    Hours start where the weather data's hours start (e.g. at xx:41 for PVGIS data), so each day's last hour reaches
    past 24:00; that part is drawn again at the start of the next day (a second, shifted copy of the image, clipped
    by the axis), leaving no gap at 0:00. The two plots sit in a row that wraps: side by side on a computer, one
    below the other on a phone.
    """
    hourly = result.hourly
    offset = hour_axis_offset(hourly.index)
    ### Rows: days, columns: hours of the day (by the hour's start).
    tables = {case: hourly.pivot_table(index=hourly.index.dayofyear, columns=hourly.index.hour, values=f"{component}_{case}", aggfunc="sum").reindex(columns=range(24)) for case in SHADING_CASES}
    first_day, last_day = tables["unobstructed"].index.min(), tables["unobstructed"].index.max()
    n_days = last_day - first_day + 1
    color_mapper = LinearColorMapper(palette=Inferno256, low=0.0, high=float(max(table.max().max() for table in tables.values())), nan_color="white")
    month_starts = pd.date_range(f"{hourly.index[0].year}-01-01", periods=12, freq="MS")
    plots = []
    for case in SHADING_CASES:
        plot = figure(title=f"{case.capitalize()}", x_range=plots[0].x_range if plots else (0, 24), y_range=plots[0].y_range if plots else (last_day + 0.5, first_day - 0.5),
                      width=width, height=height, **INACTIVE_TOOLS, x_axis_label="hour of day (local standard time)", y_axis_label="day of year")
        image = tables[case].to_numpy()
        images = [plot.image(image=[image], x=offset, y=first_day - 0.5, dw=24, dh=n_days, color_mapper=color_mapper),
                  plot.image(image=[image], x=offset - 24, y=first_day + 0.5, dw=24, dh=n_days, color_mapper=color_mapper)]   ### the part past 24:00, at the start of the next day
        plot.add_tools(HoverTool(renderers=images, tooltips=[("day of year", "$y{0}"), ("hour of day", "$x{0.0}"), ("radiation", "@image{0} Wh/m²")]))
        plot.yaxis.ticker = FixedTicker(ticks=list(month_starts.dayofyear))
        plot.yaxis.major_label_overrides = {day: name for day, name in zip(month_starts.dayofyear, MONTH_NAMES)}
        plot.xaxis.ticker = FixedTicker(ticks=list(range(0, 25, 3)))
        plot.add_layout(ColorBar(color_mapper=color_mapper, title=f"hourly {component} radiation on the panel (Wh/m²)", orientation="horizontal", height=12), "below")
        plots.append(plot)
    return Row(children=plots, styles={"flex-wrap": "wrap", "gap": "16px"}, sizing_mode="stretch_width")


def monthly_profiles_plot(result, width=250, height=200):
    """Average daily profile of the radiation on the panel per month (W/m²): total, direct and diffuse, each unobstructed (dashed) and obstructed (solid).

    One small plot per month with shared axes, in a row that wraps: the page width decides the number of columns
    (one on a phone, four on a computer). No toolbars, to keep the small plots clean; hovering still shows values.
    """
    profiles = result.monthly_daily_profiles()
    offset = hour_axis_offset(result.hourly.index) + 0.5   ### centre of each hour
    y_max = 1.05 * float(profiles[[f"total_{case}" for case in SHADING_CASES]].max().max())
    plots = []
    for month in profiles.index.get_level_values("month").unique():
        profile = profiles.loc[month]
        plot = figure(title=MONTH_NAMES[month - 1], x_range=plots[0].x_range if plots else (0, 24), y_range=plots[0].y_range if plots else (0, y_max), width=width, height=height, toolbar_location=None, **INACTIVE_TOOLS)
        lines = {}
        for component, color in COMPONENT_COLORS.items():
            for case, dash in CASE_DASHES.items():
                source = ColumnDataSource({"hour": profile.index.to_numpy() + offset, "radiation": profile[f"{component}_{case}"].to_numpy()})
                lines[component, case] = plot.line("hour", "radiation", source=source, line_width=2 if component == "total" else 1.5, line_color=color, line_dash=dash)
                plot.add_tools(HoverTool(renderers=[lines[component, case]], tooltips=[("", f"{component}, {case}"), ("hour", "@hour{0.0}"), ("radiation", "@radiation{0} W/m²")]))
        plot.xaxis.ticker = FixedTicker(ticks=list(range(0, 25, 6)))
        if not plots:
            items = [LegendItem(label=component, renderers=[lines[component, "obstructed"]]) for component in COMPONENT_COLORS] + [LegendItem(label="unobstructed", renderers=[lines["total", "unobstructed"]])]
            plot.add_layout(Legend(items=items, location="top_left", label_text_font_size="7pt", glyph_height=10, label_height=10, spacing=0, padding=4, background_fill_alpha=0.7))
        plots.append(plot)
    return Row(children=plots, styles={"flex-wrap": "wrap"}, sizing_mode="stretch_width")


def daily_bars_plot(result, width=900, height=320):
    """Radiation on the panel per day over the year (kWh/m²): one bar per day, unobstructed (light) behind obstructed (dark)."""
    daily = result.daily()
    ### Bars centred on noon of each day, one day wide minus a small gap; time stamps without time zone for the date axis.
    source = ColumnDataSource({"day": daily.index.tz_localize(None) + pd.Timedelta(hours=12), "date": daily.index.strftime("%d %b"),
                               **{case: daily[f"total_{case}"].to_numpy() for case in SHADING_CASES}})
    plot = figure(title="Radiation on the panel per day", x_axis_type="datetime", width=width, height=height, sizing_mode="scale_width", y_axis_label="kWh/m² per day", **INACTIVE_TOOLS)
    bars = {case: plot.vbar(x="day", top=case, width=pd.Timedelta(hours=20), source=source, color=CASE_BAR_COLORS[case]) for case in SHADING_CASES}
    plot.add_tools(HoverTool(renderers=[bars["unobstructed"]], tooltips=[("date", "@date"), ("unobstructed", "@unobstructed{0.00} kWh/m²"), ("obstructed", "@obstructed{0.00} kWh/m²")], mode="vline"))
    plot.y_range.start = 0
    plot.xaxis.formatter = DatetimeTickFormatter(months="%b", days="%d %b")
    plot.xgrid.visible = False
    plot.add_layout(Legend(items=[LegendItem(label=case, renderers=[bars[case]]) for case in SHADING_CASES], location="top_left", orientation="horizontal", background_fill_alpha=0.7))
    return plot


def orientation_contours(grid, case):
    """Contour lines of the orientation grid at `ORIENTATION_CONTOUR_FRACTIONS` of the best radiation: list of (fraction, azimuths, tilts)."""
    if min(grid.radiation[case].shape) < 2:
        return []
    generator = contourpy.contour_generator(grid.azimuths_deg, grid.tilts_deg, grid.radiation[case], line_type="Separate")
    best = grid.radiation[case].max()
    return [(fraction, line[:, 0], line[:, 1]) for fraction in ORIENTATION_CONTOUR_FRACTIONS for line in generator.lines(fraction * best) if len(line) > 1]


def orientation_plots(result, width=450, height=300):
    """Annual radiation on the panel over panel azimuth and tilt (kWh/m²), unobstructed and obstructed with one colour scale; the best orientation and the panel's are marked.

    Dotted contour lines enclose the orientations within 95 % and 90 % of each plot's best. The two plots sit in a row that wraps (side by side on a computer, one below the other on a phone).
    """
    grid = result.orientation_grid
    azimuth_step = grid.azimuths_deg[1] - grid.azimuths_deg[0] if len(grid.azimuths_deg) > 1 else 1.0
    tilt_step = grid.tilts_deg[1] - grid.tilts_deg[0] if len(grid.tilts_deg) > 1 else 1.0
    high = float(grid.radiation["unobstructed"].max())
    color_mapper = LinearColorMapper(palette=Inferno256, low=0.0, high=high)
    plots = []
    for case in SHADING_CASES:
        best = grid.best(case)
        plot = figure(title=f"{case.capitalize()}: best tilt {best['tilt_deg']:g}°, azimuth {best['azimuth_deg']:g}°, {best['radiation_kwh_m2']:.0f} kWh/m²",
                      x_range=plots[0].x_range if plots else (0, 360), y_range=plots[0].y_range if plots else (0, 90), width=width, height=height, sizing_mode="scale_width",
                      x_axis_label="panel azimuth (°, compass: 90 = east, 180 = south)", y_axis_label="panel tilt (°)", **INACTIVE_TOOLS)
        image = plot.image(image=[grid.radiation[case]], x=grid.azimuths_deg[0] - azimuth_step / 2, y=grid.tilts_deg[0] - tilt_step / 2, dw=grid.azimuths_deg[-1] - grid.azimuths_deg[0] + azimuth_step,
                           dh=grid.tilts_deg[-1] - grid.tilts_deg[0] + tilt_step, color_mapper=color_mapper)
        plot.add_tools(HoverTool(renderers=[image], tooltips=[("azimuth", "$x{0}°"), ("tilt", "$y{0}°"), ("annual radiation", "@image{0} kWh/m²")]))
        for fraction, x, y in orientation_contours(grid, case):
            plot.line(x, y, line_color="white", line_dash="dotted", line_width=1.5)
            plot.text([x[len(x) // 2]], [y[len(y) // 2]], text=[f"{fraction:.0%}"], text_color="white", text_font_size="8pt", text_align="center", text_baseline="bottom")
        markers = [plot.scatter([best["azimuth_deg"]], [best["tilt_deg"]], marker="star", size=16, fill_color=BEST_MARKER_COLOR, line_color="black")]
        labels = ["best"]
        plot.scatter([result.azimuth_deg], [result.tilt_deg], marker="circle", size=10, fill_color=PANEL_MARKER_COLOR, line_color="black")
        markers.append(plot.renderers[-1])
        labels.append("panel")
        plot.add_layout(Legend(items=[LegendItem(label=label, renderers=[marker]) for label, marker in zip(labels, markers)], location="top_right", background_fill_alpha=0.7, label_text_font_size="8pt"))
        plot.xaxis.ticker = FixedTicker(ticks=list(range(0, 361, 45)))
        plot.add_layout(ColorBar(color_mapper=color_mapper, title="annual radiation on the panel (kWh/m²)", orientation="horizontal", height=12), "below")
        plots.append(plot)
    return Row(children=plots, styles={"flex-wrap": "wrap", "gap": "16px"}, sizing_mode="stretch_width")


def result_plots(result, obstructed_sky, irradiation):
    """All plots of a yield result by name, in display order; each value is a Bokeh model for `json_item` or `save`."""
    return {"sky_hemisphere": sky_hemisphere_plot(obstructed_sky, irradiation), "monthly_profiles": monthly_profiles_plot(result), "daily_bars": daily_bars_plot(result),
            "carpet": carpet_plots(result), "orientation": orientation_plots(result)}
