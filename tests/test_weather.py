"""Tests of the weather sources with a synthetic TMY3 file, the sample PVGIS file and synthetic data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pytest

from conftest import SYNTHETIC_SITE
from pv_yield_estimator.core.weather import WeatherData, detect_weather_source, distance_km, hourly_typical_year, load_weather, resolve_site
from pv_yield_estimator.core.weather_download import open_meteo_url, parse_site_search, pvgis_tmy_url, site_search_url, weather_download_candidates


@pytest.fixture(scope="module")
def weather(synthetic_tmy3_text):
    """Synthetic clear-sky TMY3 weather data."""
    return load_weather(synthetic_tmy3_text, source="tmy3")


def test_tmy3_site_and_columns(weather, synthetic_tmy3_text):
    """Site coordinates come from the header; the radiation columns are read unchanged."""
    assert (weather.latitude, weather.longitude, weather.altitude, weather.name) == (48.0, 7.85, 274.0, "Freiburg synthetic")
    first_values = synthetic_tmy3_text.splitlines()[2].split(",")[2:]
    assert list(weather.hourly.iloc[0][["ghi", "dni", "dhi"]]) == [float(value) for value in first_values]


def test_tmy3_time_stamps_are_hour_starts_in_local_standard_time(weather):
    """The index starts at 00:00 on January 1 (UTC+1) and runs hourly; the first data row (01:00) covers 00:00–01:00."""
    index = weather.hourly.index
    assert len(index) == 8760
    assert index[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC+01:00") and index[-1] == pd.Timestamp("2025-12-31 23:00", tz="UTC+01:00")
    ### The GHI-weighted mean of the hour midpoints is near solar noon: 12:00 + (15° - 7.85°) × 4 min/° ≈ 12:29 local standard time.
    ghi = weather.hourly["ghi"]
    mean_hour = np.average(index.hour + 0.5, weights=ghi)
    assert mean_hour == pytest.approx(12.0 + (15.0 - 7.85) * 4 / 60, abs=0.15)


def test_hourly_typical_year_from_fine_multi_year_data():
    """15-minute data over two years is averaged to hours and over the years; February 29 is dropped."""
    index = pd.date_range("2023-01-01", "2024-12-31 23:45", freq="15min", tz="UTC")
    values = np.where(index.year == 2023, 100.0, 300.0) + np.where(index.minute == 0, 40.0, 0.0)
    typical = hourly_typical_year(pd.DataFrame({"ghi": values}, index=index), year=2025)
    assert len(typical) == 8760
    assert typical.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    np.testing.assert_allclose(typical["ghi"], 210.0)


def test_weather_data_validation():
    """Missing columns, naive time stamps and gaps raise errors."""
    hourly = pd.DataFrame({"ghi": [1.0, 2.0, 3.0], "dhi": 0.0, "dni": 0.0}, index=pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"))
    WeatherData(hourly)
    with pytest.raises(ValueError, match="columns"):
        WeatherData(hourly[["ghi"]])
    with pytest.raises(ValueError, match="timezone-aware"):
        WeatherData(hourly.tz_localize(None))
    with pytest.raises(ValueError, match="contiguous hourly"):
        WeatherData(hourly.iloc[[0, 2]])
    with pytest.raises(ValueError, match="Unknown weather source"):
        load_weather("", source="nowhere")


def test_resolve_site(weather, caplog):
    """Config coordinates win over those of the weather data, with the distance logged and a warning if it is large; missing ones raise."""
    caplog.set_level("INFO")
    assert resolve_site(weather) == {"latitude": 48.0, "longitude": 7.85, "altitude": 274.0}
    assert resolve_site(weather, latitude=48.1, longitude=7.85, name="x")["latitude"] == 48.1
    assert "11.1 km away" in caplog.text and "WARNING" not in caplog.text
    assert resolve_site(weather, latitude=47.0, longitude=7.85)["latitude"] == 47.0
    assert "wrong site or weather file" in caplog.text
    no_site = WeatherData(weather.hourly)
    with pytest.raises(ValueError, match="site.latitude"):
        resolve_site(no_site)


PVGIS_PATH = Path(__file__).resolve().parents[1] / "data" / "Freiburg-pvgis-tmy.csv"


def test_pvgis_tmy_sums_site_and_time_stamps():
    """The PVGIS TMY loads with its annual sums and site; hours start at the time stamp + irradiance time offset - 30 min, in UTC+1."""
    weather = load_weather(PVGIS_PATH.read_text(encoding="utf-8"), source="pvgis_tmy")
    annual = weather.hourly.sum() / 1000.0
    assert annual["ghi"] == pytest.approx(1178.293, abs=0.01) and annual["dhi"] == pytest.approx(587.177, abs=0.01) and annual["dni"] == pytest.approx(1134.549, abs=0.01)
    assert (weather.latitude, weather.longitude, weather.altitude) == (48.0, 7.85, 274.0)
    index = weather.hourly.index
    assert len(index) == 8760 and str(index.tz) == "UTC+01:00"
    assert index[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC") + pd.Timedelta(hours=0.1795 - 0.5)
    assert (weather.hourly >= 0).all().all()
    ghi = weather.hourly["ghi"]
    mean_hour = np.average(index.hour + index.minute / 60 + 0.5, weights=ghi)
    assert mean_hour == pytest.approx(12.0 + (15.0 - 7.85) * 4 / 60, abs=0.15)


def test_distance_km():
    """One degree of latitude is about 111 km; Freiburg–Berlin is about 640 km."""
    assert distance_km(48.0, 7.85, 49.0, 7.85) == pytest.approx(111.2, abs=0.1)
    assert distance_km(47.99, 7.85, 52.52, 13.40) == pytest.approx(640, abs=10)


def open_meteo_response(years=(2023, 2024), utc_offset_seconds=0):
    """A synthetic Open-Meteo archive response: clear-sky radiation for Freiburg, each value the mean of the hour *before* its time stamp."""
    hour_starts = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h", tz="UTC")
    solar_position = pvlib.solarposition.get_solarposition(hour_starts + pd.Timedelta(minutes=30), **SYNTHETIC_SITE)
    clear_sky = pvlib.clearsky.simplified_solis(solar_position["apparent_elevation"].clip(lower=0.0))
    sun_up = (solar_position["apparent_elevation"] > 0).to_numpy()
    dni, dhi = np.where(sun_up, clear_sky["dni"], 0.0), np.where(sun_up, clear_sky["dhi"], 0.0)
    ghi = dhi + dni * np.cos(np.radians(solar_position["apparent_zenith"].to_numpy())).clip(min=0.0)
    time_stamps = hour_starts + pd.Timedelta(hours=1) + pd.Timedelta(seconds=utc_offset_seconds)
    return json.dumps({"latitude": 48.0, "longitude": 7.875, "utc_offset_seconds": utc_offset_seconds, "timezone": "GMT", "elevation": 280.0,
                       "hourly": {"time": [f"{stamp:%Y-%m-%dT%H:%M}" for stamp in time_stamps], "shortwave_radiation": np.round(ghi, 1).tolist(),
                                  "diffuse_radiation": np.round(dhi, 1).tolist(), "direct_normal_irradiance": np.round(dni, 1).tolist()}})


def test_open_meteo_typical_year_and_time_stamps(weather):
    """Two years of Open-Meteo data become one typical year in local standard time, with hours starting one hour before the time stamps; the clear sky matches the synthetic TMY3."""
    content = open_meteo_response()
    assert detect_weather_source(content) == "open_meteo"
    open_meteo = load_weather(content)
    index = open_meteo.hourly.index
    assert len(index) == 8760 and str(index.tz) == "UTC+01:00" and index[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC+01:00")
    assert (open_meteo.latitude, open_meteo.longitude, open_meteo.altitude) == (48.0, 7.875, 280.0)
    assert "2023–2024" in open_meteo.name
    np.testing.assert_allclose(open_meteo.hourly.sum(), weather.hourly.sum(), rtol=0.01)
    mean_hour = np.average(index.hour + 0.5, weights=open_meteo.hourly["ghi"])
    assert mean_hour == pytest.approx(12.0 + (15.0 - 7.85) * 4 / 60, abs=0.15)
    ### Time stamps in another time zone are converted back with utc_offset_seconds.
    shifted = load_weather(open_meteo_response(utc_offset_seconds=7200), source="open_meteo")
    np.testing.assert_allclose(shifted.hourly.to_numpy(), open_meteo.hourly.to_numpy())


def test_open_meteo_errors():
    """Error responses and gaps give messages naming the problem."""
    with pytest.raises(ValueError, match="Latitude must be in range"):
        load_weather(json.dumps({"error": True, "reason": "Latitude must be in range of -90 to 90°."}), source="open_meteo")
    data = json.loads(open_meteo_response(years=(2024,)))
    data["hourly"]["shortwave_radiation"][100] = None
    with pytest.raises(ValueError, match="1 hours without values"):
        load_weather(json.dumps(data), source="open_meteo")


def test_detect_weather_source(synthetic_tmy3_text):
    """PVGIS CSV starts with "Latitude", PVGIS JSON has "outputs", everything else is read as TMY3."""
    assert detect_weather_source(PVGIS_PATH.read_text(encoding="utf-8")) == "pvgis_tmy"
    assert detect_weather_source('{"inputs": {}, "outputs": {"tmy_hourly": []}}') == "pvgis_tmy"
    assert detect_weather_source(synthetic_tmy3_text) == "tmy3"
    assert load_weather(PVGIS_PATH.read_text(encoding="utf-8")).source == "pvgis_tmy"


def test_weather_download_candidates():
    """Automatic choice: PVGIS first, then Open-Meteo; in the browser only the CORS-enabled Open-Meteo; URLs carry the site and years."""
    candidates = weather_download_candidates(48.0, 7.85, n_years=5, end_year=2024)
    assert [candidate["service"] for candidate in candidates] == ["pvgis_tmy", "open_meteo"]
    assert candidates[0]["url"] == pvgis_tmy_url(48.0, 7.85) and "lat=48.0000&lon=7.8500&outputformat=csv&usehorizon=1" in candidates[0]["url"]
    assert candidates[1]["filename"] == "weather_open_meteo_48.000_7.850_2020-2024.json" and candidates[1]["weather_source"] == "open_meteo"
    assert "start_date=2020-01-01&end_date=2024-12-31" in candidates[1]["url"] and "direct_normal_irradiance" in candidates[1]["url"]
    browser = weather_download_candidates(48.0, 7.85, in_browser=True, source="auto")
    assert [candidate["service"] for candidate in browser] == ["open_meteo"]
    assert "models=era5" in open_meteo_url(48.0, 7.85, open_meteo_model="era5")
    with pytest.raises(ValueError, match="no CORS"):
        weather_download_candidates(48.0, 7.85, download_service="pvgis_tmy", in_browser=True)
    with pytest.raises(ValueError, match="site.latitude"):
        weather_download_candidates(None, 7.85)
    with pytest.raises(ValueError, match="Unknown weather download service"):
        weather_download_candidates(48.0, 7.85, download_service="meteonorm")


def test_site_search():
    """The site search URL encodes the query; results parse into names and coordinates."""
    assert site_search_url("Freiburg im Breisgau, Hauptstraße 1") == "https://nominatim.openstreetmap.org/search?q=Freiburg+im+Breisgau%2C+Hauptstra%C3%9Fe+1&format=jsonv2&limit=5"
    places = parse_site_search('[{"display_name": "Freiburg im Breisgau, Baden-Württemberg, Deutschland", "lat": "47.9960901", "lon": "7.8494005"}]')
    assert places == [{"name": "Freiburg im Breisgau, Baden-Württemberg, Deutschland", "latitude": 47.9960901, "longitude": 7.8494005}]
    with pytest.raises(ValueError, match="empty"):
        site_search_url("  ")
