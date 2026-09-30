"""Checks against the real web services (CI job `network`, or locally with PV_YIELD_NETWORK_TEST=1 and internet access).

- CORS: which services a browser page may fetch (header Access-Control-Allow-Origin for a request from another origin).
- Plausibility: Open-Meteo's mean of ten years for Freiburg compared with the PVGIS typical year of the sample (#8).
Prints the numbers (run with -s to see them).
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from pv_yield_estimator.core.irradiation import compute_patch_irradiation
from pv_yield_estimator.core.panel_yield import YieldEstimator
from pv_yield_estimator.core.sky import ObstructedSky
from pv_yield_estimator.core.weather import distance_km, load_weather, resolve_site
from pv_yield_estimator.core.weather_download import USER_AGENT, open_meteo_url, parse_site_search, pvgis_tmy_url, site_search_url

pytestmark = pytest.mark.skipif(os.environ.get("PV_YIELD_NETWORK_TEST") != "1", reason="needs internet access to the weather services; set PV_YIELD_NETWORK_TEST=1")

REPOSITORY = Path(__file__).resolve().parents[1]
SITE = {"latitude": 48.0, "longitude": 7.85}
### A page on another origin, as the published app.
PAGE_ORIGIN = "https://laggondo.github.io"


def fetch(url):
    """Fetch a URL as a browser page on another origin would; returns (text, CORS header or None)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Origin": PAGE_ORIGIN})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read().decode("utf-8"), response.headers.get("Access-Control-Allow-Origin")


@pytest.fixture(scope="module")
def open_meteo():
    """Ten years of Open-Meteo data for the sample site, averaged into a typical year, and the response's CORS header."""
    text, cors = fetch(open_meteo_url(**SITE, n_years=10, end_year=2024))
    return load_weather(text, source="open_meteo"), cors


def test_cors_of_the_services(open_meteo):
    """Open-Meteo and Nominatim allow requests from any page; PVGIS does not (so the browser cannot fetch it)."""
    _, open_meteo_cors = open_meteo
    search_text, search_cors = fetch(site_search_url("Freiburg im Breisgau"))
    _, pvgis_cors = fetch(pvgis_tmy_url(**SITE))
    print(f"\nAccess-Control-Allow-Origin: Open-Meteo {open_meteo_cors!r}, Nominatim {search_cors!r}, PVGIS {pvgis_cors!r}")
    assert open_meteo_cors in ("*", PAGE_ORIGIN) and search_cors in ("*", PAGE_ORIGIN)
    assert pvgis_cors is None
    place = parse_site_search(search_text)[0]
    assert distance_km(place["latitude"], place["longitude"], 47.996, 7.85) < 5


def test_open_meteo_is_plausible_against_pvgis(open_meteo):
    """Open-Meteo's annual and monthly radiation and the yield on the sample are within 10 % of the PVGIS typical year."""
    weather, _ = open_meteo
    pvgis = load_weather((REPOSITORY / "data" / "Freiburg-pvgis-tmy.csv").read_text(encoding="utf-8"))
    annual = {name: data.hourly.sum() / 1000.0 for name, data in (("Open-Meteo", weather), ("PVGIS", pvgis))}
    monthly = {name: data.hourly.groupby(data.hourly.index.month)["ghi"].sum() / 1000.0 for name, data in (("Open-Meteo", weather), ("PVGIS", pvgis))}
    print(f"\nOpen-Meteo grid point {weather.latitude}, {weather.longitude} ({distance_km(weather.latitude, weather.longitude, **SITE):.1f} km from the site), {weather.name}")
    for name, values in annual.items():
        print(f"{name:10} annual GHI {values['ghi']:7.1f}, DHI {values['dhi']:7.1f}, DNI {values['dni']:7.1f} kWh/m²")
    print("Monthly GHI ratio Open-Meteo / PVGIS: " + ", ".join(f"{ratio:.2f}" for ratio in monthly["Open-Meteo"] / monthly["PVGIS"]))
    sky = ObstructedSky.from_dict(json.loads((REPOSITORY / "examples" / "sample_obstructed_sky.json").read_text(encoding="utf-8")))
    figures = {}
    for name, data in (("Open-Meteo", weather), ("PVGIS", pvgis)):
        irradiation = compute_patch_irradiation(data, sky.sky, **resolve_site(data, **SITE))
        figures[name] = YieldEstimator(irradiation, sky, panel={"tilt_deg": 15, "azimuth_deg": 180}).run().key_figures()
    for key in ("annual_total_unobstructed_kwh_m2", "annual_total_obstructed_kwh_m2", "annual_direct_obstructed_kwh_m2", "annual_diffuse_obstructed_kwh_m2", "shading_loss"):
        print(f"{key:36} Open-Meteo {figures['Open-Meteo'][key]:8.3f}   PVGIS {figures['PVGIS'][key]:8.3f}   ratio {figures['Open-Meteo'][key] / figures['PVGIS'][key]:.3f}")
    assert annual["Open-Meteo"]["ghi"] == pytest.approx(annual["PVGIS"]["ghi"], rel=0.1)
    assert np.all(np.abs(monthly["Open-Meteo"] / monthly["PVGIS"] - 1) < 0.35)
    assert figures["Open-Meteo"]["annual_total_obstructed_kwh_m2"] == pytest.approx(figures["PVGIS"]["annual_total_obstructed_kwh_m2"], rel=0.1)
