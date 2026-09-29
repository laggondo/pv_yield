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
    """Each carpet image is hour of day × day of year and sums to the annual radiation on the panel."""
    _, _, result = pipeline
    layout = carpet_plots(result)
    assert len(renderers_of(layout, Image)) == 2
    for plot, case in zip(layout.children, ("unobstructed", "obstructed")):
        (renderer,) = renderers_of(plot, Image)
        assert case in plot.title.text
        image = np.asarray(renderer.data_source.data["image"][0])
        assert image.shape == (24, 365)
        assert np.nansum(image) == pytest.approx(result.hourly[f"total_{case}"].sum())
    json.dumps(json_item(layout))


def test_monthly_profiles_and_all_plots(pipeline):
    """Twelve monthly plots with an unobstructed and an obstructed line each; all plots serialize."""
    irradiation, obstructed_sky, result = pipeline
    profiles = monthly_profiles_plot(result)
    assert len(list(profiles.select({"type": GlyphRenderer}))) == 24
    plots = result_plots(result, obstructed_sky, irradiation)
    assert list(plots) == ["sky_hemisphere", "monthly_profiles", "carpet"]
    for plot in plots.values():
        json.dumps(json_item(plot))
