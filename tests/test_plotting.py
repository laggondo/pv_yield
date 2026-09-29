"""Tests of the interactive Bokeh plots."""

import json
from pathlib import Path

import numpy as np
import pytest
from bokeh.embed import json_item
from bokeh.models import GlyphRenderer, Image, Patches

from pv_yield_estimator.core.irradiation import compute_patch_irradiation, sun_path_directions
from pv_yield_estimator.core.panel_yield import YieldEstimator
from pv_yield_estimator.core.sky import ObstructedSky, SkyDiscretization
from pv_yield_estimator.core.weather import load_weather, resolve_site
from pv_yield_estimator.plotting.interactive import annual_radiation_per_patch, carpet_plots, monthly_profiles_plot, polar_plot_xy, result_plots, sky_hemisphere_plot

PVGIS_PATH = Path(__file__).resolve().parents[1] / "data" / "Freiburg-pvgis-tmy.csv"


@pytest.fixture(scope="module")
def pipeline():
    """Irradiation per sky patch for the sample weather on a coarse sky, the southern half obstructed below 30° elevation, and the yield result."""
    weather = load_weather(PVGIS_PATH.read_text(encoding="utf-8"), source="pvgis_tmy")
    sky = SkyDiscretization.from_node_count(200)
    irradiation = compute_patch_irradiation(weather, sky, **resolve_site(weather))
    centers = sky.patch_centers()
    obstructed_sky = ObstructedSky(sky, (centers[:, 1] < 0) & (centers[:, 2] < np.sin(np.radians(30))))
    result = YieldEstimator(irradiation, obstructed_sky, panel={"tilt_deg": 30.0}).run()
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
    """One patch per sky patch with its obstruction flag; works without irradiation, too; serializable for the browser."""
    irradiation, obstructed_sky, _ = pipeline
    for plot, with_radiation in ((sky_hemisphere_plot(obstructed_sky, irradiation), True), (sky_hemisphere_plot(obstructed_sky), False)):
        (patches,) = renderers_of(plot, Patches)
        data = patches.data_source.data
        assert len(data["xs"]) == obstructed_sky.sky.n_patches
        assert list(data["obstructed"]).count("yes") == np.count_nonzero(obstructed_sky.obstructed)
        assert ("total_kwh_m2" in data) == with_radiation
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
    for plot in [sky_hemisphere_plot(obstructed_sky, irradiation), *carpet_plots(result).children, *monthly_profiles_plot(result).children]:
        assert plot.toolbar.active_drag is None and plot.toolbar.active_scroll is None


def test_monthly_profiles_and_all_plots(pipeline):
    """Twelve monthly plots in a wrapping row, each with total, direct and diffuse lines, unobstructed and obstructed; all plots serialize."""
    irradiation, obstructed_sky, result = pipeline
    profiles = monthly_profiles_plot(result)
    assert len(profiles.children) == 12 and profiles.styles["flex-wrap"] == "wrap"
    assert len(list(profiles.select({"type": GlyphRenderer}))) == 12 * 6
    plots = result_plots(result, obstructed_sky, irradiation)
    assert list(plots) == ["sky_hemisphere", "monthly_profiles", "carpet"]
    for plot in plots.values():
        json.dumps(json_item(plot))
