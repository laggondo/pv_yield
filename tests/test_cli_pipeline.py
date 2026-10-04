"""Tests of the CLI subcommands on the sample data; downloads are replaced by canned responses."""

import json
from pathlib import Path

import pandas as pd
import pytest

from pv_yield_estimator.cli import main as cli
from pv_yield_estimator.cli.main import main
from test_weather import open_meteo_response

REPOSITORY = Path(__file__).resolve().parents[1]
POINT_CLOUD = REPOSITORY / "data" / "2026-06-01_22-25-29_red_red.csv"
WEATHER = REPOSITORY / "data" / "Freiburg-pvgis-tmy.csv"
SAMPLE_CONFIG = REPOSITORY / "examples" / "sample_config.yaml"
SAMPLE_SKY = REPOSITORY / "examples" / "sample_obstructed_sky.json"
### Coarse orientation grid and no PNG/PDF output, to keep the tests fast where those are not tested.
FAST = ["-m", "orientation.tilt_step_deg=10", "orientation.azimuth_step_deg=30", "output.plots_png=false", "output.pdf_report=false"]


def key_figures(directory):
    """Key figures from a results directory."""
    return json.loads((directory / "results.json").read_text())["key_figures"]


def test_step_by_step_equals_run(tmp_path):
    """The separate steps give the same key figures as `run`; `run` writes all result files; overrides reach the consumers."""
    main(["obstruction", str(POINT_CLOUD), "-o", str(tmp_path / "sky.json"), "-c", str(SAMPLE_CONFIG)])
    main(["irradiation", str(WEATHER), "--sky", str(tmp_path / "sky.json"), "-o", str(tmp_path / "irradiation.json"), "-c", str(SAMPLE_CONFIG)])
    main(["-c", str(SAMPLE_CONFIG), "yield", str(tmp_path / "irradiation.json"), str(tmp_path / "sky.json"), "-d", str(tmp_path / "steps"), *FAST])
    main(["run", "-s", str(POINT_CLOUD), "-w", str(WEATHER), "-d", str(tmp_path / "run"), "-c", str(SAMPLE_CONFIG), "-m", "orientation.tilt_step_deg=10", "orientation.azimuth_step_deg=30"])
    assert key_figures(tmp_path / "steps") == pytest.approx(key_figures(tmp_path / "run"), rel=1e-5)
    results = json.loads((tmp_path / "steps" / "results.json").read_text())
    assert results["config"]["panel"]["tilt_deg"] == 15 and results["key_figures"]["sky_view_factor_horizontal"] == pytest.approx(0.5500, abs=1e-4)
    for name in ("obstructed_sky.json", "patch_irradiation.json", "results.json", "config.yaml", "hourly.csv", "monthly.csv", "orientation_grid.csv", "plots.html", "report.pdf", "plot_carpet.png"):
        assert (tmp_path / "run" / name).is_file(), name
    assert not (tmp_path / "steps" / "report.pdf").exists() and not (tmp_path / "steps" / "plot_carpet.png").exists()
    assert len(pd.read_csv(tmp_path / "run" / "hourly.csv", comment="#")) == 8760
    assert "Annual radiation per sky patch" in (tmp_path / "run" / "plots.html").read_text(encoding="utf-8")
    main(["yield", str(tmp_path / "irradiation.json"), str(tmp_path / "sky.json"), "-d", str(tmp_path / "tilt30"), "-c", str(SAMPLE_CONFIG), *FAST, "panel.tilt_deg=30"])
    tilt30 = json.loads((tmp_path / "tilt30" / "results.json").read_text())
    assert tilt30["config"]["panel"]["tilt_deg"] == 30
    assert tilt30["key_figures"]["annual_diffuse_unobstructed_kwh_m2"] < results["key_figures"]["annual_diffuse_unobstructed_kwh_m2"]


def test_run_from_sky_description_with_optimized_orientation(tmp_path):
    """`run` also takes an obstructed sky description; an unset tilt is optimized."""
    main(["run", "-s", str(SAMPLE_SKY), "-w", str(WEATHER), "-d", str(tmp_path), *FAST, "panel.tilt_deg=null", "panel.azimuth_deg=180"])
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["optimized_angles"] == ["tilt_deg"] and results["key_figures"]["tilt_deg"] in range(0, 91, 10)
    assert json.loads((tmp_path / "obstructed_sky.json").read_text())["obstructed"] == json.loads(SAMPLE_SKY.read_text())["obstructed"]


def test_without_obstruction(tmp_path):
    """Without an obstructed sky description, `yield` and `run` compute a free sky: no shading loss, sky view factor 1."""
    main(["irradiation", str(WEATHER), "-o", str(tmp_path / "irradiation.json"), "-m", "simulation.n_sky_nodes=200"])
    main(["yield", str(tmp_path / "irradiation.json"), "-d", str(tmp_path / "yield"), *FAST])
    main(["run", "-w", str(WEATHER), "-d", str(tmp_path / "run"), *FAST, "simulation.n_sky_nodes=200"])
    for directory in ("yield", "run"):
        results = json.loads((tmp_path / directory / "results.json").read_text())
        assert results["key_figures"]["shading_loss"] == pytest.approx(0.0, abs=1e-12) and results["key_figures"]["sky_view_factor_horizontal"] == pytest.approx(1.0)
        assert results["inputs"]["obstructed_sky"]["method"] == "none"
    assert key_figures(tmp_path / "yield") == pytest.approx(key_figures(tmp_path / "run"), rel=1e-6)


def test_yield_rejects_mismatched_sky(tmp_path):
    """Irradiation computed on another discretization than the obstruction raises a helpful error."""
    main(["obstruction", str(POINT_CLOUD), "-o", str(tmp_path / "sky.json"), "-m", "simulation.n_sky_nodes=100"])
    main(["irradiation", str(WEATHER), "-o", str(tmp_path / "irradiation.json"), "-m", "simulation.n_sky_nodes=200"])
    with pytest.raises(ValueError, match="different sky discretizations"):
        main(["yield", str(tmp_path / "irradiation.json"), str(tmp_path / "sky.json")])


def test_missing_input_files_are_named(tmp_path):
    """A missing input file raises an error naming it and how to produce it."""
    with pytest.raises(FileNotFoundError, match="irradiation per sky patch .*irradiation.json.*`irradiation` subcommand"):
        main(["yield", str(tmp_path / "irradiation.json"), str(tmp_path / "sky.json")])
    with pytest.raises(FileNotFoundError, match="weather file .*`weather` subcommand"):
        main(["irradiation", str(tmp_path / "weather.csv"), "-o", str(tmp_path / "irradiation.json")])


@pytest.fixture
def fake_web(monkeypatch):
    """Canned answers instead of downloads: the site search finds Freiburg, PVGIS answers with the sample file (or fails, if `pvgis_fails` is set), Open-Meteo with synthetic data. Returns the list of requested URLs."""
    requests = []
    state = {"pvgis_fails": False}

    def fetch_text(url, timeout=180):
        """Answer a request by its host."""
        requests.append(url)
        if "nominatim" in url:
            return '[{"display_name": "Freiburg im Breisgau, Deutschland", "lat": "47.996", "lon": "7.849"}]'
        if "re.jrc.ec.europa.eu" in url:
            if state["pvgis_fails"]:
                raise RuntimeError(f'HTTP 400 from {url}: {{"message":"Location over the sea"}}')
            return WEATHER.read_text(encoding="utf-8")
        if "open-meteo" in url:
            return open_meteo_response(years=(2024,))
        raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(cli, "fetch_text", fetch_text)
    return requests, state


def test_weather_download_with_site_search_and_fallback(tmp_path, monkeypatch, fake_web):
    """`weather` looks up the site, downloads PVGIS; if PVGIS fails, Open-Meteo; `run` without a weather file downloads once and reuses the file."""
    requests, state = fake_web
    monkeypatch.chdir(tmp_path)
    main(["weather", "-m", "site.query=Freiburg"])
    assert "nominatim" in requests[0] and "lat=47.9960&lon=7.8490" in requests[1]
    assert (tmp_path / "weather_pvgis_tmy_47.996_7.849.csv").read_text(encoding="utf-8") == WEATHER.read_text(encoding="utf-8")
    state["pvgis_fails"] = True
    main(["weather", "-o", str(tmp_path / "fallback.json"), "-m", "site.latitude=48.0", "site.longitude=7.85", "weather.end_year=2024", "weather.n_years=1"])
    assert "open-meteo" in requests[-1] and json.loads((tmp_path / "fallback.json").read_text())["latitude"] == 48.0
    requests.clear()
    arguments = ["run", "-s", str(SAMPLE_SKY), "-d", str(tmp_path / "run"), *FAST, "site.latitude=48.0", "site.longitude=7.85", "weather.download_service=open_meteo", "weather.end_year=2024", "weather.n_years=1"]
    main(arguments)
    assert len(requests) == 1 and (tmp_path / "run" / "weather_open_meteo_48.000_7.850_2024-2024.json").is_file()
    main(arguments)
    assert len(requests) == 1
    assert json.loads((tmp_path / "run" / "results.json").read_text())["inputs"]["weather"]["weather_source"] == "open_meteo"
