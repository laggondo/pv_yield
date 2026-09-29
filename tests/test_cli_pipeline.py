"""Tests of the CLI subcommands on the sample data."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pv_yield_estimator.cli.main import main

REPOSITORY = Path(__file__).resolve().parents[1]
POINT_CLOUD = REPOSITORY / "data" / "2026-06-01_22-25-29_red_red.csv"
WEATHER = REPOSITORY / "data" / "Freiburg-pvgis-tmy.csv"
SAMPLE_CONFIG = REPOSITORY / "examples" / "sample_config.yaml"


def test_step_by_step_equals_run(tmp_path):
    """The separate steps give the same key figures as `run`, and overrides reach the consumers."""
    main(["obstruction", str(POINT_CLOUD), "-o", str(tmp_path / "sky.json"), "-c", str(SAMPLE_CONFIG)])
    main(["irradiation", str(WEATHER), "--sky", str(tmp_path / "sky.json"), "-o", str(tmp_path / "irradiation.json"), "-c", str(SAMPLE_CONFIG)])
    main(["-c", str(SAMPLE_CONFIG), "yield", str(tmp_path / "irradiation.json"), str(tmp_path / "sky.json"), "-o", str(tmp_path / "results.yaml"), "--hourly-csv", str(tmp_path / "hourly.csv")])
    main(["run", str(POINT_CLOUD), str(WEATHER), "-d", str(tmp_path / "run"), "-c", str(SAMPLE_CONFIG)])
    step_by_step = yaml.safe_load((tmp_path / "results.yaml").read_text())
    run = yaml.safe_load((tmp_path / "run" / "key_figures.yaml").read_text())
    assert step_by_step["key_figures"] == pytest.approx(run["key_figures"], rel=1e-5)
    assert step_by_step["config"]["panel"]["tilt_deg"] == 15
    assert step_by_step["key_figures"]["sky_view_factor_horizontal"] == pytest.approx(0.5500, abs=1e-4)
    assert len(pd.read_csv(tmp_path / "hourly.csv")) == 8760
    assert "Sky hemisphere" in (tmp_path / "run" / "plots.html").read_text(encoding="utf-8")
    main(["yield", str(tmp_path / "irradiation.json"), str(tmp_path / "sky.json"), "-o", str(tmp_path / "tilt30.yaml"), "-c", str(SAMPLE_CONFIG), "-m", "panel.tilt_deg=30"])
    tilt30 = yaml.safe_load((tmp_path / "tilt30.yaml").read_text())
    assert tilt30["config"]["panel"]["tilt_deg"] == 30
    assert tilt30["key_figures"]["annual_diffuse_unobstructed_kwh_m2"] < step_by_step["key_figures"]["annual_diffuse_unobstructed_kwh_m2"]


def test_yield_rejects_mismatched_sky(tmp_path):
    """Irradiation computed on another discretization than the obstruction raises a helpful error."""
    main(["obstruction", str(POINT_CLOUD), "-o", str(tmp_path / "sky.json"), "-m", "simulation.n_sky_nodes=100"])
    main(["irradiation", str(WEATHER), "-o", str(tmp_path / "irradiation.json"), "-m", "simulation.n_sky_nodes=200", "weather.source=pvgis_tmy"])
    with pytest.raises(ValueError, match="different sky discretizations"):
        main(["yield", str(tmp_path / "irradiation.json"), str(tmp_path / "sky.json")])
