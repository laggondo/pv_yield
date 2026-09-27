"""Proof of concept for the platform (#6) and plotting (#7) decisions: pvlib sun positions and a Bokeh sky plot.

Runs in the browser via Pyodide (loaded by index.html) and natively: `python web/proof/proof.py` writes a standalone
HTML file with BokehJS inlined, as the CLI would. Throwaway code, not part of the package.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import pvlib
from bokeh.embed import json_item
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Inferno256
from bokeh.plotting import figure

LATITUDE, LONGITUDE = 47.99, 7.84   ### Freiburg, as the sample weather file in data/


def upper_hemisphere_triangles(subdivisions=4):
    """Triangulate the upper sky hemisphere by subdividing the upper half of an octahedron; returns unit-vector nodes and index triples."""
    corners = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    triangles = np.array([[corners[i], corners[(i + 1) % 4], corners[4]] for i in range(4)])
    for _ in range(subdivisions):
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, bc, ca = [m / np.linalg.norm(m, axis=1, keepdims=True) for m in (a + b, b + c, c + a)]
        triangles = np.concatenate([np.stack(t, axis=1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    nodes, indices = np.unique(np.round(triangles.reshape(-1, 3), 12), axis=0, return_inverse=True)
    return nodes, indices.reshape(-1, 3)


def to_plot_xy(vectors):
    """Project unit vectors (x east, y north, z up) to an equidistant polar plot: radius = zenith angle in degrees, north up, east right."""
    zenith_deg = np.degrees(np.arccos(np.clip(vectors[..., 2], -1, 1)))
    azimuth_rad = np.arctan2(vectors[..., 0], vectors[..., 1])
    return zenith_deg * np.sin(azimuth_rad), zenith_deg * np.cos(azimuth_rad)


def sun_vectors(times):
    """Sun positions from pvlib as unit vectors (x east, y north, z up), with the solar position table."""
    position = pvlib.solarposition.get_solarposition(times, LATITUDE, LONGITUDE)
    elevation, azimuth = np.radians(position["apparent_elevation"].to_numpy()), np.radians(position["azimuth"].to_numpy())
    return np.stack([np.cos(elevation) * np.sin(azimuth), np.cos(elevation) * np.cos(azimuth), np.sin(elevation)], axis=1), position


def make_sky_plot(step_minutes=10):
    """Sky hemisphere with triangular patches colored by annual clear-sky direct irradiation, a fake obstruction and sun paths; returns the figure and timings."""
    timings = {}
    start = time.perf_counter()
    nodes, triangles = upper_hemisphere_triangles()
    timings["sky discretization"] = time.perf_counter() - start

    start = time.perf_counter()
    times = pd.date_range("2025-01-01", "2026-01-01", freq=f"{step_minutes}min", tz="Etc/GMT-1", inclusive="left")
    suns, position = sun_vectors(times)
    timings[f"pvlib sun positions ({len(times)} time steps)"] = time.perf_counter() - start

    start = time.perf_counter()
    above_horizon = suns[:, 2] > 0
    direct_normal = pvlib.clearsky.simplified_solis(position["apparent_elevation"].to_numpy()[above_horizon])["dni"]
    centers = nodes[triangles].mean(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    patch_of_sun = np.argmax(suns[above_horizon] @ centers.T, axis=1)   ### nearest patch center; good enough for the proof
    irradiation_kwh = np.bincount(patch_of_sun, weights=np.asarray(direct_normal) * step_minutes / 60 / 1000, minlength=len(triangles))
    timings["irradiation per patch"] = time.perf_counter() - start

    center_elevation_deg = np.degrees(np.arcsin(centers[:, 2]))
    center_azimuth_deg = np.degrees(np.arctan2(centers[:, 0], centers[:, 1])) % 360
    obstructed = (center_elevation_deg < 25) & (center_azimuth_deg > 200) & (center_azimuth_deg < 280)   ### fake building to the west-south-west, drawn hatched

    patch_x, patch_y = to_plot_xy(nodes[triangles])
    source = ColumnDataSource(dict(
        xs=list(patch_x), ys=list(patch_y), irradiation_kwh=irradiation_kwh, elevation_deg=center_elevation_deg, azimuth_deg=center_azimuth_deg,
        obstructed=np.where(obstructed, "yes", "no"), hatch=np.where(obstructed, "x", " ")))
    color_mapper = LinearColorMapper(palette=Inferno256, low=0.01, high=float(irradiation_kwh.max()), low_color="#f4f4f4")
    plot = figure(title="Sky hemisphere: annual clear-sky direct irradiation per patch (kWh/m²), Freiburg", x_range=(-100, 100), y_range=(-100, 100),
                  width=640, height=560, sizing_mode="scale_width", tools="pan,wheel_zoom,box_zoom,reset,save,tap", active_scroll="wheel_zoom", x_axis_label="east ← → (zenith angle, °)", y_axis_label="north ↑ (zenith angle, °)")
    patches = plot.patches("xs", "ys", source=source, fill_color={"field": "irradiation_kwh", "transform": color_mapper}, line_color="#999999", line_width=0.3,
                            hatch_pattern="hatch", hatch_color="#1f77b4", hatch_alpha=0.8)
    plot.add_tools(HoverTool(renderers=[patches], tooltips=[("elevation", "@elevation_deg{0.0}°"), ("azimuth", "@azimuth_deg{0.0}°"), ("irradiation", "@irradiation_kwh{0.0} kWh/m²"), ("obstructed", "@obstructed")]))
    plot.add_layout(ColorBar(color_mapper=color_mapper, title="kWh/m²"), "right")

    for day, color in (("2025-06-21", "#2ca02c"), ("2025-03-20", "#17becf"), ("2025-12-21", "#9467bd")):
        day_times = pd.date_range(day, periods=24 * 12, freq="5min", tz="Etc/GMT-1")
        day_suns, _ = sun_vectors(day_times)
        x, y = to_plot_xy(day_suns[day_suns[:, 2] > 0])
        plot.line(x, y, line_color=color, line_width=2, legend_label=f"sun path {day}")
    angle = np.linspace(0, 2 * np.pi, 361)
    plot.line(90 * np.sin(angle), 90 * np.cos(angle), line_color="black")
    plot.text([0, 93, 0, -93], [93, 0, -93, 0], text=["N", "E", "S", "W"], text_align="center", text_baseline="middle")
    plot.legend.location = "bottom_right"
    return plot, timings


def sky_plot_json():
    """Build the sky plot and return it as JSON for Bokeh.embed.embed_item, plus timings and versions (called from index.html)."""
    plot, timings = make_sky_plot()
    import bokeh
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "pvlib": pvlib.__version__, "bokeh": bokeh.__version__}
    return json.dumps({"plot": json_item(plot), "timings": timings, "versions": versions})


if __name__ == "__main__":
    from bokeh.io import save
    from bokeh.resources import INLINE
    output_path = sys.argv[1] if len(sys.argv) > 1 else "sky_plot.html"
    plot, timings = make_sky_plot()
    save(plot, output_path, resources=INLINE, title="Sky plot proof")
    print(f"Wrote {output_path}; timings in seconds: {json.dumps(timings, indent=1)}")
