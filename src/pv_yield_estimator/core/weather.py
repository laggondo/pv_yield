"""Weather sources and the internal hourly weather data representation."""

import datetime
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
    """Common interface of weather sources (TMY3 file, PVGIS TMY, later automatic downloads).

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


class PvgisTmyWeatherSource(WeatherSource):
    """Typical meteorological year from PVGIS (the EU's free solar data service), CSV or JSON as downloaded.

    PVGIS time stamps are UTC. The irradiance values are the satellite estimate at the time stamp plus the file's
    "irradiance time offset" (e.g. 0.18 h), so each value is taken as the mean of the hour centred there. The index is
    converted to local standard time, UTC + `utc_offset_hours` (default: longitude / 15° rounded, i.e. UTC+1 for
    Central Europe), so days and hours of the day group like local time. The year is set to `year` for all rows, as
    a TMY mixes months from different years; a February 29 is dropped.
    """

    source_name = "pvgis_tmy"

    def __init__(self, content, year=2025, utc_offset_hours=None, pvgis_format="csv", **kwargs):
        self.content = content
        self.year = year
        self.utc_offset_hours = utc_offset_hours
        self.pvgis_format = pvgis_format

    def load(self):
        """Parse the PVGIS content with pvlib and convert it to the internal representation."""
        ### pvlib's PVGIS CSV parser reads bytes.
        buffer = io.BytesIO(self.content.encode("utf-8")) if isinstance(self.content, str) and self.pvgis_format == "csv" else io.StringIO(self.content) if isinstance(self.content, str) else self.content
        data, metadata = pvlib.iotools.read_pvgis_tmy(buffer, pvgis_format=self.pvgis_format, map_variables=True)
        inputs = metadata["inputs"]
        latitude, longitude = float(inputs["latitude"]), float(inputs["longitude"])
        hourly = data[list(RADIATION_COLUMNS)].astype(float).clip(lower=0.0)
        index = hourly.index
        hourly = hourly[~((index.month == 2) & (index.day == 29))]
        index = hourly.index
        hourly.index = pd.DatetimeIndex(pd.to_datetime({"year": self.year, "month": index.month, "day": index.day, "hour": index.hour})).tz_localize("UTC")
        hourly = hourly.sort_index()
        time_offset_hours = float(inputs.get("irradiance time offset", 0.0))
        hourly.index = hourly.index + pd.Timedelta(hours=time_offset_hours - 0.5)
        utc_offset_hours = round(longitude / 15.0) if self.utc_offset_hours is None else self.utc_offset_hours
        hourly.index = hourly.index.tz_convert(datetime.timezone(datetime.timedelta(hours=utc_offset_hours)))
        if len(hourly) != 8760:
            log.warning(f"PVGIS TMY has {len(hourly)} hours, expected 8760")
        weather = WeatherData(hourly=hourly, latitude=latitude, longitude=longitude, altitude=float(inputs.get("elevation", 0.0)), name=f"PVGIS TMY {latitude:.3f}, {longitude:.3f}", source=self.source_name, metadata=metadata)
        log.info(f"PVGIS TMY weather at {latitude}°, {longitude}° (irradiance time offset {time_offset_hours:g} h, local standard time UTC{utc_offset_hours:+g}): annual GHI {hourly['ghi'].sum() / 1000:.1f}, DHI {hourly['dhi'].sum() / 1000:.1f}, DNI {hourly['dni'].sum() / 1000:.1f} kWh/m²")
        return weather


### Weather sources by name, as selected by `weather.source` in the config; further sources plug in here.
WEATHER_SOURCES = {Tmy3WeatherSource.source_name: Tmy3WeatherSource, PvgisTmyWeatherSource.source_name: PvgisTmyWeatherSource}


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
