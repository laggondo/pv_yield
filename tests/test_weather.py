"""Tests of the weather sources with a synthetic TMY3 file, the sample PVGIS file and synthetic data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pv_yield_estimator.core.weather import WeatherData, hourly_typical_year, load_weather, resolve_site


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
    """Config coordinates win over those of the weather data, with a warning if they differ; missing ones raise."""
    assert resolve_site(weather) == {"latitude": 48.0, "longitude": 7.85, "altitude": 274.0}
    assert resolve_site(weather, latitude=47.0, name="x")["latitude"] == 47.0
    assert "differs" in caplog.text
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
