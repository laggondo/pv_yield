"""Tests of the result export (JSON, YAML, CSV) and the PDF report."""

import io
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pv_yield_estimator.config import config_from_yaml, default_config, merge_configs
from pv_yield_estimator.core.export import RESULTS_FORMAT_VERSION, export_files
from pv_yield_estimator.core.irradiation import compute_patch_irradiation
from pv_yield_estimator.core.panel_yield import YieldEstimator
from pv_yield_estimator.core.sky import ObstructedSky
from pv_yield_estimator.core.weather import load_weather, resolve_site
from pv_yield_estimator.plotting.report import pdf_report

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pipeline():
    """Config, site, irradiation, obstructed sky and result for the sample data, with the orientation optimized."""
    config = merge_configs(default_config(), {"panel": {"azimuth_deg": 180}, "orientation": {"tilt_step_deg": 5, "azimuth_step_deg": 10, "compare": [[90, 180]]}})
    obstructed_sky = ObstructedSky.from_dict(json.loads((REPOSITORY / "examples" / "sample_obstructed_sky.json").read_text(encoding="utf-8")))
    weather = load_weather((REPOSITORY / "data" / "Freiburg-pvgis-tmy.csv").read_text(encoding="utf-8"))
    site = resolve_site(weather, **config["site"])
    irradiation = compute_patch_irradiation(weather, obstructed_sky.sky, **site, **config["simulation"])
    result = YieldEstimator(irradiation, obstructed_sky, **config).run()
    return config, site, irradiation, obstructed_sky, result


def test_export_files(pipeline):
    """results.json holds config, key figures, monthly values and the comparison; config.yaml reloads; the CSV files hold the tables."""
    config, site, irradiation, obstructed_sky, result = pipeline
    files = export_files(result, config, site, irradiation, obstructed_sky)
    assert set(files) == {"results.json", "config.yaml", "hourly.csv", "daily.csv", "monthly.csv", "orientation_comparison.csv", "orientation_grid.csv"}
    results = json.loads(files["results.json"])
    assert results["format_version"] == RESULTS_FORMAT_VERSION and results["kind"] == "results"
    assert results["config"] == config and results["site"] == site
    assert results["key_figures"] == pytest.approx(result.key_figures())
    assert results["optimized_angles"] == ["tilt_deg"] and results["inputs"]["obstructed_sky"]["method"] == "lidar"
    assert len(results["monthly_daily_average"]) == 12 and [row["label"] for row in results["orientation"]["comparison"]][-1] == "compare"
    assert config_from_yaml(files["config.yaml"]) == config
    for name in ("hourly", "daily", "monthly", "orientation_comparison", "orientation_grid"):
        assert files[f"{name}.csv"].startswith(f"# pv_yield_estimator") and f"format_version {RESULTS_FORMAT_VERSION}" in files[f"{name}.csv"].splitlines()[0]
    hourly = pd.read_csv(io.StringIO(files["hourly.csv"]), comment="#")
    assert len(hourly) == 8760 and hourly["total_obstructed"].sum() / 1000 == pytest.approx(results["key_figures"]["annual_total_obstructed_kwh_m2"], rel=1e-5)
    assert len(pd.read_csv(io.StringIO(files["daily.csv"]), comment="#")) == 365
    grid = pd.read_csv(io.StringIO(files["orientation_grid.csv"]), comment="#")
    assert len(grid) == 19 * 37 and grid["radiation_obstructed_kwh_m2"].max() == pytest.approx(results["orientation"]["best_obstructed"]["radiation_kwh_m2"], abs=1e-3)
    yaml.safe_load(files["config.yaml"])


def test_pdf_report(pipeline):
    """The report is a PDF with the two table pages and one page per plot."""
    config, site, irradiation, obstructed_sky, result = pipeline
    pdf = pdf_report(result, config, site, irradiation, obstructed_sky)
    assert pdf.startswith(b"%PDF")
    assert b"/Count 7" in pdf   ### page count in the PDF's page tree
