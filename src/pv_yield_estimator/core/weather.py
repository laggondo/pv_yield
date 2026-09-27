"""Weather sources and the internal hourly weather data representation."""

import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pvlib

log = logging.getLogger(__name__)

RADIATION_COLUMNS = ("ghi", "dhi", "dni")


@dataclass
class WeatherData:
    """Hourly weather data in the internal representation.

    `hourly` has the columns ghi (global horizontal), dhi (diffuse horizontal) and dni (direct normal), each the mean
    irradiance over the hour in W/m² (equal to the hour's irradiation in Wh/m²). The index holds the *start* of each
    hour as timezone-aware timestamps, contiguous and hourly; sun positions for the hour are computed within
    [start, start + 1 h). Daily and monthly values group by the index's time zone. Site coordinates are given if the
    source provides them.
    """

    hourly: pd.DataFrame
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    name: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        missing = [column for column in RADIATION_COLUMNS if column not in self.hourly]
        if missing:
            raise ValueError(f"Weather data from {self.source or 'unknown source'} lacks the columns {missing}")
        index = self.hourly.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise ValueError(f"Weather data from {self.source or 'unknown source'} needs a timezone-aware DatetimeIndex, got {type(index).__name__} (tz {getattr(index, 'tz', None)})")
        steps = np.diff(index)
        if len(index) > 1 and not (steps == pd.Timedelta(hours=1)).all():
            raise ValueError(f"Weather data from {self.source or 'unknown source'} must be contiguous hourly; first irregular step {pd.Timedelta(steps[steps != pd.Timedelta(hours=1)][0])} after {index[:-1][steps != pd.Timedelta(hours=1)][0]}")


def hourly_typical_year(radiation, year=2025):
    """Convert irradiance (W/m², any columns) of finer resolution and/or several real years to one hourly typical year.

    Finer data is averaged to hours (the mean irradiance equals the hour's irradiation in Wh/m²); several years are
    averaged per (month, day, hour), with February 29 dropped. The index must be timezone-aware and label the start
    of each interval; the result is indexed by hour starts in `year` (a non-leap year) in the same time zone.
    """
    hourly = radiation.resample("1h").mean()
    hourly = hourly[~((hourly.index.month == 2) & (hourly.index.day == 29))]
    typical = hourly.groupby([hourly.index.month, hourly.index.day, hourly.index.hour]).mean()
    index = pd.DatetimeIndex([pd.Timestamp(year=year, month=month, day=day, hour=hour) for month, day, hour in typical.index]).tz_localize(radiation.index.tz)
    typical.index = index
    return typical.sort_index()


class WeatherSource(ABC):
    """Common interface of weather sources (TMY3 file, later automatic downloads).

    Each source handles its own time stamp convention and converts to `WeatherData`: hourly, index = hour start,
    timezone-aware. The core receives file content, not paths.
    """

    source_name = "abstract"

    @abstractmethod
    def load(self):
        """Return the weather data as `WeatherData`."""


class Tmy3WeatherSource(WeatherSource):
    """Typical meteorological year in TMY3 format, e.g. from MeteoNorm.

    TMY3 time stamps are local standard time (no daylight saving) and label the *end* of the hour the values cover
    (01:00 covers 00:00–01:00, 24:00 the last hour of the day). They are shifted to hour starts; the year is set to
    `year` for all rows, as a TMY mixes months from different years.
    """

    source_name = "tmy3"

    def __init__(self, content, year=2025, **kwargs):
        self.content = content
        self.year = year

    def load(self):
        """Parse the TMY3 content with pvlib and convert it to the internal representation."""
        data, metadata = pvlib.iotools.read_tmy3(io.StringIO(self.content) if isinstance(self.content, str) else self.content, coerce_year=self.year, map_variables=True)
        hourly = data[list(RADIATION_COLUMNS)].astype(float)
        hourly.index = hourly.index - pd.Timedelta(hours=1)
        if len(hourly) != 8760:
            log.warning(f"TMY3 file has {len(hourly)} hours, expected 8760")
        weather = WeatherData(hourly=hourly, latitude=float(metadata["latitude"]), longitude=float(metadata["longitude"]), altitude=float(metadata["altitude"]), name=str(metadata.get("Name", "")), source=self.source_name, metadata={key: value for key, value in metadata.items()})
        log.info(f"TMY3 weather {weather.name!r} at {weather.latitude}°, {weather.longitude}° (UTC{metadata['TZ']:+g}): annual GHI {hourly['ghi'].sum() / 1000:.1f}, DHI {hourly['dhi'].sum() / 1000:.1f}, DNI {hourly['dni'].sum() / 1000:.1f} kWh/m²")
        return weather


### Weather sources by name, as selected by `weather.source` in the config; further sources plug in here.
WEATHER_SOURCES = {Tmy3WeatherSource.source_name: Tmy3WeatherSource}


def load_weather(content, source="tmy3", **kwargs):
    """Load weather data from file content with the source named in the config's weather section."""
    if source not in WEATHER_SOURCES:
        raise ValueError(f"Unknown weather source {source!r}; known: {', '.join(WEATHER_SOURCES)}")
    return WEATHER_SOURCES[source](content, **kwargs).load()


def resolve_site(weather, latitude=None, longitude=None, altitude=None, **kwargs):
    """Site coordinates: from the config's site section if given, else from the weather data.

    Warns if both are given and differ by more than 0.1°.
    """
    site = {"latitude": latitude, "longitude": longitude, "altitude": altitude}
    for key in site:
        from_weather = getattr(weather, key)
        if site[key] is None:
            site[key] = from_weather
        elif from_weather is not None and key != "altitude" and abs(site[key] - from_weather) > 0.1:
            log.warning(f"Site {key} {site[key]} from the config differs from the weather data's {from_weather}; using the config value")
    if site["latitude"] is None or site["longitude"] is None:
        raise ValueError(f"Site coordinates unknown: the weather source {weather.source!r} provides none; set site.latitude and site.longitude in the config")
    if site["altitude"] is None:
        site["altitude"] = 0.0
    return site
