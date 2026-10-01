"""Tests of the interactive Bokeh plots and their static matplotlib versions."""

import json
from pathlib import Path

import numpy as np
import pytest
from bokeh.embed import json_item
from bokeh.models import GlyphRenderer, Image, MultiLine, Patches, VBar

from pv_yield_estimator.core.irradiation import compute_patch_irradiation, sun_path_directions
from pv_yield_estimator.core.panel_yield import YieldEstimator
from pv_yield_estimator.core.sky import ObstructedSky, SkyDiscretization
from pv_yield_estimator.core.weather import load_weather, resolve_site
from pv_yield_estimator.plotting.interactive import annual_radiation_per_patch, carpet_plots, daily_bars_plot, monthly_profiles_plot, orientation_contours, orientation_plots, polar_plot_xy, result_plots, sky_hemisphere_plot
from pv_yield_estimator.plotting.static import static_figures

PVGIS_PATH = Path(__file__).resolve().parents[1] / "data" / "Freiburg-pvgis-tmy.csv"


@pytest.fixture(scope="module")
def pipeline():
    """Irradiation per sky patch for the sample weather on a coarse sky, the southern half obstructed below 30° elevation, and the yield result."""
    weather = load_weather(PVGIS_PATH.read_text(encoding="utf-8"), source="pvgis_tmy")
    sky = SkyDiscretization.from_node_count(200)
    irradiation = compute_patch_irradiation(weather, sky, **resolve_site(weather))
    centers = sky.patch_centers()
    obstructed_sky = ObstructedSky(sky, (centers[:, 1] < 0) & (centers[:, 2] < np.sin(np.radians(30))))
    result = YieldEstimator(irradiation, obstructed_sky, panel={"tilt_deg": 30.0, "azimuth_deg": 180.0}, orientation={"tilt_step_deg": 5, "azimuth_step_deg": 10}).run()
    return irradiation, obstructed_sky, result


def renderers_of(model, glyph_type):
    """All glyph renderers with the given glyph type within a Bokeh model (figure or layout)."""
    return [renderer for renderer in model.select({"type": GlyphRenderer}) if isinstance(renderer.glyph, glyph_type)]


def test_polar_projection():
    """Zenith maps to the centre, the horizon to radius 90 with north up and east right."""
    x, y = polar_plot_xy(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(x, [0.0, 0.0, 90.0], atol=1e-12)
    np.testing.assert_allclose(y, [0.0, 90.0, 0.0], atol=1e-12)


def test_annual_radiation_per_patch_sums(pipeline):
    """Direct parts sum to the annual direct normal irradiation; diffuse parts to DHI × 2π / π = 2 × DHI (the whole hemisphere at normal incidence)."""
    irradiation, _, _ = pipeline
    radiation = annual_radiation_per_patch(irradiation)
    assert radiation["direct"].sum() == pytest.approx(irradiation.annual_direct_normal_per_patch().sum() / 1000)
    assert radiation["diffuse"].sum() == pytest.approx(2 * irradiation.dhi.sum() / 1000)
    np.testing.assert_allclose(radiation["total"], radiation["direct"] + radiation["diffuse"])


def test_sky_hemisphere_plot(pipeline):
    """One patch per sky patch with its obstruction flag, obstructed patches dimmed (with radiation) and the border outlined; works without irradiation, too; serializable for the browser."""
    irradiation, obstructed_sky, _ = pipeline
    n_obstructed = np.count_nonzero(obstructed_sky.obstructed)
    for plot, with_radiation in ((sky_hemisphere_plot(obstructed_sky, irradiation), True), (sky_hemisphere_plot(obstructed_sky), False)):
        patches, *veil = renderers_of(plot, Patches)
        data = patches.data_source.data
        assert len(data["xs"]) == obstructed_sky.sky.n_patches
        assert list(data["obstructed"]).count("yes") == n_obstructed
        assert ("total_kwh_m2" in data) == with_radiation
        assert [len(renderer.data_source.data["xs"]) for renderer in veil] == ([n_obstructed] if with_radiation else [])
        (outline,) = renderers_of(plot, MultiLine)
        assert len(outline.data_source.data["xs"]) == len(obstructed_sky.boundary_edges()) > 0
        json.dumps(json_item(plot))


def test_sun_path_reaches_expected_noon_elevation():
    """On 21 June at 48° N, the sun culminates at about 90° − 48° + 23.4° ≈ 65.4° elevation, due south."""
    directions = sun_path_directions("2025-06-21", 48.0, 7.85, timezone="UTC+01:00", step_minutes=2)
    highest = directions[np.argmax(directions[:, 2])]
    assert np.degrees(np.arcsin(highest[2])) == pytest.approx(65.4, abs=0.3)
    assert highest[1] < 0 and abs(highest[0]) < 0.02
    assert np.all(directions[:, 2] > 0)


def test_carpet_plots_hold_the_hourly_values(pipeline):
    """Each carpet plot shows day of year (downwards) × hour of day, sums to the annual radiation on the panel, and covers 0–24 h without a gap."""
    _, _, result = pipeline
    layout = carpet_plots(result)
    assert layout.styles["flex-wrap"] == "wrap"
    offset = result.hourly.index[0].minute / 60 + result.hourly.index[0].second / 3600
    assert offset > 0.5   ### PVGIS hours start at xx:41, so the last hour of a day reaches past 24:00
    for plot, case in zip(layout.children, ("unobstructed", "obstructed")):
        wrapped, main = sorted(renderers_of(plot, Image), key=lambda renderer: renderer.glyph.x)
        assert case in plot.title.text.lower()
        assert plot.y_range.start > plot.y_range.end   ### 1 January at the top
        assert (plot.x_range.start, plot.x_range.end) == (0, 24)
        image = np.asarray(main.data_source.data["image"][0])
        assert image.shape == (365, 24)
        assert np.nansum(image) == pytest.approx(result.hourly[f"total_{case}"].sum())
        ### The main image starts at the first hour start; the shifted copy (one day later, 24 h earlier) fills 0:00 up to there.
        assert main.glyph.x == pytest.approx(offset) and main.glyph.dw == 24
        assert wrapped.glyph.x + wrapped.glyph.dw == pytest.approx(offset) and wrapped.glyph.y == main.glyph.y + 1
    json.dumps(json_item(layout))


def test_plots_leave_touch_and_wheel_to_the_page(pipeline):
    """No drag or scroll tool is active by default, so swiping over a plot on a phone scrolls the page (checked in the browser smoke test)."""
    irradiation, obstructed_sky, result = pipeline
    for plot in [sky_hemisphere_plot(obstructed_sky, irradiation), *carpet_plots(result).children, *monthly_profiles_plot(result).children, daily_bars_plot(result), *orientation_plots(result).children]:
        assert plot.toolbar.active_drag is None and plot.toolbar.active_scroll is None


def test_monthly_profiles_and_all_plots(pipeline):
    """Twelve monthly plots in a wrapping row, each with total, direct and diffuse lines, unobstructed and obstructed; all plots serialize."""
    irradiation, obstructed_sky, result = pipeline
    profiles = monthly_profiles_plot(result)
    assert len(profiles.children) == 12 and profiles.styles["flex-wrap"] == "wrap"
    assert len(list(profiles.select({"type": GlyphRenderer}))) == 12 * 6
    plots = result_plots(result, obstructed_sky, irradiation)
    assert list(plots) == ["sky_hemisphere", "monthly_profiles", "daily_bars", "carpet", "orientation"]
    for plot in plots.values():
        json.dumps(json_item(plot))


def test_daily_bars(pipeline):
    """One bar per day and case, holding the daily radiation on the panel."""
    _, _, result = pipeline
    plot = daily_bars_plot(result)
    bars = renderers_of(plot, VBar)
    assert len(bars) == 2
    data = bars[0].data_source.data
    assert len(data["day"]) == 365
    np.testing.assert_allclose(data["obstructed"], result.daily()["total_obstructed"])
    assert np.all(data["obstructed"] <= data["unobstructed"] + 1e-9)


def test_orientation_plots(pipeline):
    """Two heatmaps (tilt × azimuth) with the grid's values; contours at 95 % and 90 % of the best lie below the best value."""
    _, _, result = pipeline
    layout = orientation_plots(result)
    assert len(layout.children) == 2 and layout.styles["flex-wrap"] == "wrap"
    grid = result.orientation_grid
    for plot, case in zip(layout.children, ("unobstructed", "obstructed")):
        (image,) = renderers_of(plot, Image)
        np.testing.assert_allclose(image.data_source.data["image"][0], grid.radiation[case])
        assert f"/ {grid.best(case)['azimuth_deg']:g}°" in plot.title.text and plot.sizing_mode is None   ### fixed size, so the row wraps on a phone
    contours = orientation_contours(grid, "unobstructed")
    assert {fraction for fraction, _, _ in contours} == {0.95, 0.9}
    json.dumps(json_item(layout))


def test_static_figures(pipeline, tmp_path):
    """The static figures render to PNG and PDF without pyplot."""
    irradiation, obstructed_sky, result = pipeline
    figures = static_figures(result, obstructed_sky, irradiation)
    assert list(figures) == list(result_plots(result, obstructed_sky, irradiation))
    for name, figure in figures.items():
        figure.savefig(tmp_path / f"{name}.png", dpi=50)
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
    figures["carpet"].savefig(tmp_path / "carpet.pdf")
    assert (tmp_path / "carpet.pdf").read_bytes().startswith(b"%PDF")
