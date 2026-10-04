"""Tests of the orientation comparison and optimization on the sample data."""

import json
from pathlib import Path

import numpy as np
import pytest

from pv_yield_estimator.core.irradiation import compute_patch_irradiation
from pv_yield_estimator.core.orientation import orientation_grid
from pv_yield_estimator.core.panel_yield import YieldEstimator, compute_panel_radiation
from pv_yield_estimator.core.sky import ObstructedSky
from pv_yield_estimator.core.weather import load_weather, resolve_site

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def obstructed_sky():
    """The sample obstructed sky description."""
    return ObstructedSky.from_dict(json.loads((REPOSITORY / "examples" / "sample_obstructed_sky.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def irradiation(obstructed_sky):
    """Irradiation per sky patch of the sample weather on the sample sky discretization."""
    weather = load_weather((REPOSITORY / "data" / "Freiburg-pvgis-tmy.csv").read_text(encoding="utf-8"))
    return compute_patch_irradiation(weather, obstructed_sky.sky, **resolve_site(weather))


@pytest.mark.parametrize("tilt_deg, azimuth_deg", [(0, 180), (30, 180), (60, 90), (90, 270), (45, 0)])
def test_grid_matches_exact_hourly_sum(irradiation, obstructed_sky, tilt_deg, azimuth_deg):
    """The grid's annual sums per patch give the annual radiation within 0.5 % of the exact hourly computation."""
    grid = orientation_grid(irradiation, obstructed_sky, fixed_tilt_deg=tilt_deg, fixed_azimuth_deg=azimuth_deg)
    exact = compute_panel_radiation(irradiation, obstructed_sky, tilt_deg, azimuth_deg).sum() / 1000.0
    for case in ("unobstructed", "obstructed"):
        assert grid.radiation[case][0, 0] == pytest.approx(exact[f"total_{case}"], rel=5e-3)


def test_grid_shape_and_best_orientation(irradiation, obstructed_sky):
    """The grid spans tilt 0–90° and azimuth 0–360°; unobstructed, the best orientation in Freiburg faces south at about 35–40° tilt."""
    grid = orientation_grid(irradiation, obstructed_sky, tilt_step_deg=5, azimuth_step_deg=10)
    assert grid.radiation["obstructed"].shape == (19, 37)
    np.testing.assert_allclose(grid.radiation["obstructed"][:, 0], grid.radiation["obstructed"][:, -1])
    assert np.all(grid.radiation["obstructed"] <= grid.radiation["unobstructed"] + 1e-9)
    best = grid.best("unobstructed")
    assert 30 <= best["tilt_deg"] <= 45 and 170 <= best["azimuth_deg"] <= 190
    assert best["radiation_kwh_m2"] == pytest.approx(grid.radiation["unobstructed"].max())
    exported = grid.to_dict()
    assert len(exported["radiation_obstructed_kwh_m2"]) == 19 and exported["best_unobstructed"] == best


def test_unset_angles_are_optimized(irradiation, obstructed_sky):
    """An unset tilt or azimuth is optimized, the other one kept; the result is at least as good as the neighbouring grid orientations."""
    settings = {"tilt_step_deg": 2, "azimuth_step_deg": 5}
    full = YieldEstimator(irradiation, obstructed_sky, panel={"tilt_deg": None}, orientation=settings).run()
    assert full.optimized_angles == ["tilt_deg", "azimuth_deg"]
    grid = full.orientation_grid
    assert full.tilt_deg == grid.best("obstructed")["tilt_deg"] and full.azimuth_deg == grid.best("obstructed")["azimuth_deg"]
    fixed_azimuth = YieldEstimator(irradiation, obstructed_sky, panel={"azimuth_deg": 90.0}, orientation=settings).run()
    assert fixed_azimuth.azimuth_deg == 90.0 and fixed_azimuth.optimized_angles == ["tilt_deg"]
    assert fixed_azimuth.tilt_deg == grid.tilts_deg[np.argmax(grid.radiation["obstructed"][:, list(grid.azimuths_deg).index(90.0)])]
    fixed = YieldEstimator(irradiation, obstructed_sky, panel={"tilt_deg": 15, "azimuth_deg": 180}).run()
    assert fixed.optimized_angles == [] and (fixed.tilt_deg, fixed.azimuth_deg) == (15, 180)
    assert full.key_figures()["annual_total_obstructed_kwh_m2"] > fixed.key_figures()["annual_total_obstructed_kwh_m2"]


def test_comparison_table(irradiation, obstructed_sky):
    """The comparison lists the panel, the best orientations and the requested ones, without duplicates, with exact key figures."""
    panel = {"tilt_deg": 15, "azimuth_deg": 180, "performance_ratio": 0.8}
    result = YieldEstimator(irradiation, obstructed_sky, panel=panel, orientation={"tilt_step_deg": 5, "azimuth_step_deg": 10, "compare": [[90, 180], [15, 180], [30, 90]]}).run()
    rows = result.orientation_comparison
    assert [row["label"] for row in rows] == ["panel", "best obstructed", "best unobstructed", "compare", "compare"]
    assert rows[0]["annual_total_obstructed_kwh_m2"] == pytest.approx(result.key_figures()["annual_total_obstructed_kwh_m2"])
    assert (rows[3]["tilt_deg"], rows[3]["azimuth_deg"]) == (90.0, 180.0)
    assert rows[1]["annual_total_obstructed_kwh_m2"] == max(row["annual_total_obstructed_kwh_m2"] for row in rows)
    with pytest.raises(ValueError, match="tilt_deg, azimuth_deg"):
        YieldEstimator(irradiation, obstructed_sky, panel=panel, orientation={"compare": [[30]]}).run()
