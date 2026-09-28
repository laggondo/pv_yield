"""Tests of the irradiation per sky patch and the radiation on the panel, yield and key figures."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pv_yield_estimator.core.irradiation import PatchIrradiation, compute_patch_irradiation, sub_steps_per_hour_for
from pv_yield_estimator.core.panel_yield import YieldEstimator, compute_panel_radiation
from pv_yield_estimator.core.sky import ObstructedSky, SkyDiscretization, directions_from_zenith_azimuth
from pv_yield_estimator.core.weather import load_weather, resolve_site
from pv_yield_estimator.file_format import to_json_text

TMY3_PATH = Path(__file__).resolve().parents[1] / "data" / "Freiburg-hour.csv"


@pytest.fixture(scope="module")
def sky():
    """The legacy default resolution."""
    return SkyDiscretization.from_node_count(500)


@pytest.fixture(scope="module")
def weather():
    """The sample TMY3 weather data."""
    return load_weather(TMY3_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def irradiation(weather, sky):
    """Irradiation per sky patch for the sample weather data."""
    return compute_patch_irradiation(weather, sky, **resolve_site(weather))


def unobstructed(sky):
    """Obstructed sky description without obstructions."""
    return ObstructedSky(sky, np.zeros(sky.n_patches, dtype=bool))


def test_sub_steps_keep_sun_steps_below_half_a_patch(sky):
    """The automatic sub-step count keeps the sun's step (≤ 15°/h) below half the mean patch edge."""
    n_sub_steps = sub_steps_per_hour_for(sky)
    assert 15.0 / n_sub_steps <= 0.5 * np.degrees(sky.mean_edge_angle_rad())
    assert 15.0 / (n_sub_steps - 1) > 0.5 * np.degrees(sky.mean_edge_angle_rad())
    assert sub_steps_per_hour_for(sky, sub_steps_per_hour=12) == 12


def test_direct_vectors_reproduce_input_dni(irradiation, weather):
    """Per hour, the lengths of the direct vectors sum to the hour's DNI (up to hours with the sun never up).

    |V| is slightly smaller than the sum of the sub-steps' irradiation, as the sun directions within a patch differ a little.
    """
    per_hour = np.bincount(irradiation.direct_hours, weights=np.linalg.norm(irradiation.direct_vectors, axis=1), minlength=irradiation.n_hours)
    dni = weather.hourly["dni"].to_numpy()
    assert per_hour.sum() == pytest.approx(dni.sum(), rel=2e-4)
    has_sun = per_hour > 0
    np.testing.assert_allclose(per_hour[has_sun], dni[has_sun], rtol=1e-3)
    np.testing.assert_allclose(irradiation.annual_direct_normal_per_patch().sum(), per_hour.sum())
    assert irradiation.annual_direct_vectors().shape == (irradiation.sky.n_patches, 3)
    assert np.all(irradiation.direct_vectors[:, 2] > 0)


def test_horizontal_unobstructed_panel_reproduces_ghi(irradiation, weather, sky):
    """A horizontal unobstructed panel receives the global horizontal radiation (annual sum within 0.5 %)."""
    hourly = compute_panel_radiation(irradiation, unobstructed(sky), tilt_deg=0.0)
    np.testing.assert_allclose(hourly["diffuse_unobstructed"], weather.hourly["dhi"], rtol=1e-12)
    assert hourly["total_unobstructed"].sum() == pytest.approx(weather.hourly["ghi"].sum(), rel=5e-3)
    assert (hourly["total_obstructed"] == hourly["total_unobstructed"]).all()


def test_sun_patches_follow_the_sun(irradiation):
    """At summer noon the direct vector points south and high; the sun path moves east to west in the morning and evening."""
    index = irradiation.index
    noon = np.nonzero(index == pd.Timestamp("2025-06-21 12:00", tz="UTC+01:00"))[0][0]
    vector = irradiation.direct_vectors[irradiation.direct_hours == noon].sum(axis=0)
    assert vector[1] < 0 and vector[2] / np.linalg.norm(vector) > np.sin(np.radians(60))
    morning = irradiation.direct_vectors[np.isin(irradiation.direct_hours, np.nonzero(index.hour == 7)[0])].sum(axis=0)
    evening = irradiation.direct_vectors[np.isin(irradiation.direct_hours, np.nonzero(index.hour == 17)[0])].sum(axis=0)
    assert morning[0] > 0 > evening[0]


def test_irradiation_json_round_trip(irradiation):
    """The irradiation per sky patch survives a JSON round trip (values rounded to 1e-4 Wh/m²)."""
    loaded = PatchIrradiation.from_dict(json.loads(to_json_text(irradiation.to_dict())), source="test.json")
    assert loaded.sky.same_as(irradiation.sky) and loaded.time_start == irradiation.time_start and (loaded.index == irradiation.index).all()
    np.testing.assert_array_equal(loaded.direct_patches, irradiation.direct_patches)
    np.testing.assert_allclose(loaded.direct_vectors, irradiation.direct_vectors, atol=1e-4)
    np.testing.assert_allclose(loaded.dhi, irradiation.dhi)
    assert loaded.metadata["sub_steps_per_hour"] == irradiation.metadata["sub_steps_per_hour"]
    with pytest.raises(ValueError, match="not an irradiation"):
        PatchIrradiation.from_dict(irradiation.to_dict() | {"kind": "obstructed_sky"})


def synthetic_irradiation(sky, sun_directions, dni=500.0, dhi=100.0):
    """One hour per sun direction, each with the given direct normal and diffuse horizontal irradiation."""
    patches = sky.find_patches(sun_directions)
    n_hours = len(sun_directions)
    return PatchIrradiation(sky, pd.Timestamp("2025-06-01", tz="UTC"), np.full(n_hours, dni + dhi), np.full(n_hours, dhi), np.full(n_hours, dni), np.arange(n_hours), patches, dni * sun_directions, 48.0, 7.85)


def test_synthetic_half_sky_obstruction(sky):
    """Obstructing the southern half of the sky blocks a southern sun and about half of the diffuse radiation."""
    sun_directions = directions_from_zenith_azimuth(np.radians([40.0, 40.0]), np.radians([180.0, 0.0]))
    irradiation = synthetic_irradiation(sky, sun_directions)
    obstructed_sky = ObstructedSky(sky, sky.patch_centers()[:, 1] < 0)
    hourly = compute_panel_radiation(irradiation, obstructed_sky, tilt_deg=0.0)
    np.testing.assert_allclose(hourly["direct_unobstructed"], 500.0 * np.cos(np.radians(40.0)))
    np.testing.assert_allclose(hourly["direct_obstructed"], [0.0, 500.0 * np.cos(np.radians(40.0))])
    np.testing.assert_allclose(hourly["diffuse_obstructed"], 50.0, rtol=0.02)
    ### A vertical panel facing north sees nothing from the southern sun and half the sky's diffuse radiation.
    vertical = compute_panel_radiation(irradiation, unobstructed(sky), tilt_deg=90.0, azimuth_deg=0.0)
    np.testing.assert_allclose(vertical["direct_unobstructed"], [0.0, 500.0 * np.sin(np.radians(40.0))])
    np.testing.assert_allclose(vertical["diffuse_unobstructed"], 50.0, rtol=1e-3)


def test_tilted_diffuse_is_isotropic_view_factor(irradiation, sky):
    """Unobstructed diffuse radiation on a tilted panel is DHI × (1 + cos tilt) / 2."""
    hourly = compute_panel_radiation(irradiation, unobstructed(sky), tilt_deg=30.0, azimuth_deg=200.0)
    np.testing.assert_allclose(hourly["diffuse_unobstructed"], irradiation.dhi * (1 + np.cos(np.radians(30.0))) / 2, rtol=2e-3)


def test_fully_obstructed_sky_and_key_figures(sky):
    """A fully obstructed sky gives zero obstructed radiation and a shading loss of 1; yields follow the panel parameters."""
    irradiation = synthetic_irradiation(sky, directions_from_zenith_azimuth(np.radians([30.0] * 48), np.radians([180.0] * 48)))
    result = YieldEstimator(irradiation, ObstructedSky(sky, np.ones(sky.n_patches, dtype=bool)), panel={"tilt_deg": 30.0, "azimuth_deg": 180.0, "area_m2": 2.0, "efficiency": 0.2, "performance_ratio": 0.8}, site={}).run()
    figures = result.key_figures()
    assert figures["annual_total_obstructed_kwh_m2"] == 0.0 and figures["shading_loss"] == 1.0 and figures["sky_view_factor"] == 0.0
    radiation = figures["annual_total_unobstructed_kwh_m2"]
    assert radiation == pytest.approx(48 * (0.5 + 0.1 * (1 + np.cos(np.radians(30.0))) / 2), rel=2e-3)
    assert figures["annual_yield_unobstructed_kwh"] == pytest.approx(radiation * 2.0 * 0.2 * 0.8)
    assert figures["specific_yield_unobstructed_kwh_kwp"] == pytest.approx(radiation * 0.8)
    assert figures["rated_power_kwp"] == pytest.approx(0.4)
    assert result.daily()["total_unobstructed"].sum() == pytest.approx(radiation)
    monthly = result.monthly_daily_average()
    assert list(monthly.index) == [6] and monthly.loc[6, "total_unobstructed"] == pytest.approx(radiation / 2)
    assert result.monthly_daily_profiles().loc[(6, 5), "total_unobstructed"] == pytest.approx(1000 * radiation / 48)


def test_mismatched_discretizations_raise(irradiation):
    """Irradiation and obstruction on different discretizations raise an error explaining the fix."""
    with pytest.raises(ValueError, match="different sky discretizations"):
        compute_panel_radiation(irradiation, unobstructed(SkyDiscretization.from_node_count(100)))


def test_pvgis_horizontal_unobstructed_matches_ghi_hourly(sky):
    """With the PVGIS time convention, DHI + direct on a horizontal panel reproduces each hour's GHI closely (checks the time alignment)."""
    weather = load_weather((Path(__file__).resolve().parents[1] / "data" / "Freiburg-pvgis-tmy.csv").read_text(encoding="utf-8"), source="pvgis_tmy")
    irradiation = compute_patch_irradiation(weather, sky, **resolve_site(weather))
    hourly = compute_panel_radiation(irradiation, unobstructed(sky), tilt_deg=0.0)
    error = hourly["total_unobstructed"].to_numpy() - weather.hourly["ghi"].to_numpy()
    assert np.sqrt(np.mean(error ** 2)) < 2.0
    assert hourly["total_unobstructed"].sum() == pytest.approx(weather.hourly["ghi"].sum(), rel=3e-3)
