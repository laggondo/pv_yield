"""Stand-in for the legacy `meteo_data_handler` module, which is not part of `legacy_code/`.

Provides just what `legacy_code/estimate_inc_rad.py` uses. Assumptions (the original is not available): TMY3 time
stamps are local standard time labelling the end of the hour; the hourly sun position is taken at the middle of the
hour, consistent with the legacy sub-steps, which span [time stamp - 1 h, time stamp].
"""

import numpy as np
import pvlib

import utilityLib as ul


class MeteoDataHandlerTMY3:
    """TMY3 weather data with the interface `estimate_inc_rad.py` expects (Duffie-Beckman sun angles)."""

    def __init__(self, weather_file_path):
        data, self.metadata = pvlib.iotools.read_tmy3(weather_file_path, coerce_year=2010, map_variables=True)
        data.index = data.index.tz_localize(None)
        self.df_weather_data = data[["dni", "dhi", "ghi"]].astype(float)

    def get_SunPosition_object(self):
        """Legacy sun position calculator; longitudes are positive west, the time zone meridian from the UTC offset."""
        return ul.SunPosition(latitude=np.radians(self.metadata["latitude"]), longitude=np.radians(-self.metadata["longitude"]), longitude_std=np.radians(-15.0 * self.metadata["TZ"]))

    def get_zen_azm(self):
        """Sun zenith and azimuth (Duffie-Beckman) in radians at the middle of each hour."""
        sun_position = self.get_SunPosition_object()
        angles = np.array([sun_position.get_azm_zen_angles(timestamp - np.timedelta64(30, "m")) for timestamp in self.df_weather_data.index])
        return angles[:, 0], angles[:, 1]
