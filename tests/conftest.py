"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pvlib
import pytest

### Site of the synthetic weather data: Freiburg, as the sample LiDAR scan.
SYNTHETIC_SITE = {"latitude": 48.0, "longitude": 7.85, "altitude": 274.0}


@pytest.fixture(scope="session")
def synthetic_tmy3_text():
    """A TMY3 file with a clear-sky year for Freiburg (simplified Solis model at the middle of each hour).

    Written like a real TMY3 file: local standard time UTC+1, each row labelled with the *end* of its hour (24:00 for
    the last hour of a day). Only the radiation columns are included.
    """
    hour_starts = pd.date_range("2025-01-01", periods=8760, freq="h", tz="UTC+01:00")
    solar_position = pvlib.solarposition.get_solarposition(hour_starts + pd.Timedelta(minutes=30), **SYNTHETIC_SITE)
    clear_sky = pvlib.clearsky.simplified_solis(solar_position["apparent_elevation"].clip(lower=0.0))
    sun_up = (solar_position["apparent_elevation"] > 0).to_numpy()
    dni, dhi = np.where(sun_up, clear_sky["dni"], 0.0), np.where(sun_up, clear_sky["dhi"], 0.0)
    ghi = dhi + dni * np.cos(np.radians(solar_position["apparent_zenith"].to_numpy())).clip(min=0.0)
    lines = ["999999,Freiburg synthetic,,1,48.000,7.850,274\n", "Date (MM/DD/YYYY),Time (HH:MM),GHI (W/m^2),DNI (W/m^2),DHI (W/m^2)\n"]
    lines += [f"{start:%m/%d}/2005,{start.hour + 1:02d}:00,{g:.1f},{n:.1f},{d:.1f}\n" for start, g, n, d in zip(hour_starts, ghi, dni, dhi)]
    return "".join(lines)
