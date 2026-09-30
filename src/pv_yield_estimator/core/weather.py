"""Weather sources and the internal hourly weather data representation."""

import datetime
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pvlib

log = logging.getLogger(__name__)

RADIATION_COLUMNS = ("ghi", "dhi", "dni")
EARTH_RADIUS_KM = 6371.0


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


def local_standard_time_zone(longitude, utc_offset_hours=None):
    """Fixed-offset time zone of local standard time (no daylight saving): UTC + `utc_offset_hours`, by default longitude / 15° rounded (UTC+1 for Central Europe)."""
    if utc_offset_hours is None:
        utc_offset_hours = round(longitude / 15.0)
    return datetime.timezone(datetime.timedelta(hours=utc_offset_hours))


def distance_km(latitude_a, longitude_a, latitude_b, longitude_b):
    """Great circle distance between two points on the earth in km (haversine formula)."""
    latitude_a, longitude_a, latitude_b, longitude_b = np.radians([latitude_a, longitude_a, latitude_b, longitude_b])
    haversine = np.sin((latitude_b - latitude_a) / 2) ** 2 + np.cos(latitude_a) * np.cos(latitude_b) * np.sin((longitude_b - longitude_a) / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine)))


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

    def __init__(self, content, year=2025, utc_offset_hours=None, pvgis_format=None, **kwargs):
        self.content = content
        self.year = year
        self.utc_offset_hours = utc_offset_hours
        ### csv or json; detected from text content if not given.
        self.pvgis_format = pvgis_format or ("json" if isinstance(content, str) and content.lstrip().startswith("{") else "csv")

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
        time_zone = local_standard_time_zone(longitude, self.utc_offset_hours)
        hourly.index = hourly.index.tz_convert(time_zone)
        if len(hourly) != 8760:
            log.warning(f"PVGIS TMY has {len(hourly)} hours, expected 8760")
        weather = WeatherData(hourly=hourly, latitude=latitude, longitude=longitude, altitude=float(inputs.get("elevation", 0.0)), name=f"PVGIS TMY {latitude:.3f}, {longitude:.3f}", source=self.source_name, metadata=metadata)
        log.info(f"PVGIS TMY weather at {latitude}°, {longitude}° (irradiance time offset {time_offset_hours:g} h, local standard time {time_zone}): annual GHI {hourly['ghi'].sum() / 1000:.1f}, DHI {hourly['dhi'].sum() / 1000:.1f}, DNI {hourly['dni'].sum() / 1000:.1f} kWh/m²")
        return weather


class OpenMeteoWeatherSource(WeatherSource):
    """Hourly historical weather from the Open-Meteo archive API (JSON response as downloaded), averaged over its years.

    Open-Meteo (free for non-commercial use, no key, CORS-enabled) serves reanalysis data (by default ERA5, about
    25 km grid) for real years. Time stamps are UTC with `timezone=GMT` (other time zones are converted back with the
    response's `utc_offset_seconds`); the radiation values are means over the *preceding* hour, so each hour starts
    one hour before its time stamp. The index is converted to local standard time (see `local_standard_time_zone`),
    and all years are averaged into one typical year (`hourly_typical_year`). Latitude, longitude and elevation are
    those of the grid cell the service used, which may lie several km from the requested site.
    """

    source_name = "open_meteo"
    ### Open-Meteo variable → internal column.
    VARIABLES = {"shortwave_radiation": "ghi", "diffuse_radiation": "dhi", "direct_normal_irradiance": "dni"}

    def __init__(self, content, year=2025, utc_offset_hours=None, **kwargs):
        self.content = content
        self.year = year
        self.utc_offset_hours = utc_offset_hours

    def load(self):
        """Parse the JSON response and convert it to the internal representation."""
        data = json.loads(self.content) if isinstance(self.content, str) else json.load(self.content)
        if not isinstance(data, dict) or "hourly" not in data:
            raise ValueError(f"Not an Open-Meteo response with hourly data: {data.get('reason') if isinstance(data, dict) else type(data).__name__!r}")
        hourly_data = data["hourly"]
        missing = [variable for variable in self.VARIABLES if variable not in hourly_data]
        if missing:
            raise ValueError(f"Open-Meteo response lacks the hourly variables {missing}; request {', '.join(self.VARIABLES)}")
        time_stamps = pd.DatetimeIndex(pd.to_datetime(hourly_data["time"])) - pd.Timedelta(seconds=data.get("utc_offset_seconds", 0))
        hour_starts = time_stamps.tz_localize("UTC") - pd.Timedelta(hours=1)
        radiation = pd.DataFrame({column: pd.to_numeric(pd.Series(hourly_data[variable], dtype=object), errors="coerce").to_numpy(dtype=float) for variable, column in self.VARIABLES.items()}, index=hour_starts)
        if radiation.isna().any().any():
            gaps = radiation.index[radiation.isna().any(axis=1)]
            raise ValueError(f"Open-Meteo data has {len(gaps)} hours without values, the first at {gaps[0]} (UTC); request complete past years only")
        latitude, longitude = float(data["latitude"]), float(data["longitude"])
        radiation = radiation.clip(lower=0.0)
        radiation.index = radiation.index.tz_convert(local_standard_time_zone(longitude, self.utc_offset_hours))
        years = sorted(set((hour_starts + pd.Timedelta(minutes=30)).year))   ### by the hours' midpoints: the first hour starts on 31 December
        hourly = hourly_typical_year(radiation, self.year)
        if len(hourly) != 8760:
            raise ValueError(f"Open-Meteo data covers {len(hourly)} hours of the year after averaging, expected 8760; request whole years")
        metadata = {key: data[key] for key in ("generationtime_ms", "timezone", "elevation", "hourly_units") if key in data} | {"first_hour_start_utc": str(hour_starts[0]), "last_hour_start_utc": str(hour_starts[-1]), "n_hours_downloaded": len(radiation)}
        weather = WeatherData(hourly=hourly, latitude=latitude, longitude=longitude, altitude=float(data.get("elevation", 0.0)), name=f"Open-Meteo {latitude:.3f}, {longitude:.3f}, mean of {years[0]}–{years[-1]}", source=self.source_name, metadata=metadata)
        log.info(f"Open-Meteo weather at the grid point {latitude}°, {longitude}°, {len(radiation)} hours averaged into a typical year (local standard time {hourly.index.tz}): annual GHI {hourly['ghi'].sum() / 1000:.1f}, DHI {hourly['dhi'].sum() / 1000:.1f}, DNI {hourly['dni'].sum() / 1000:.1f} kWh/m²")
        return weather


### Weather sources by name, as selected by `weather.source` in the config; further sources plug in here.
WEATHER_SOURCES = {source.source_name: source for source in (Tmy3WeatherSource, PvgisTmyWeatherSource, OpenMeteoWeatherSource)}


def detect_weather_source(content):
    """Name of the weather source that reads this file content: Open-Meteo or PVGIS JSON, PVGIS CSV (starts with "Latitude"), else TMY3."""
    text = content if isinstance(content, str) else ""
    start = text.lstrip()[:2000]
    if start.startswith("{"):
        return OpenMeteoWeatherSource.source_name if '"hourly"' in text else PvgisTmyWeatherSource.source_name
    if start.startswith("Latitude"):
        return PvgisTmyWeatherSource.source_name
    return Tmy3WeatherSource.source_name


def load_weather(content, source="auto", **kwargs):
    """Load weather data from file content with the source named in the config's weather section; `auto` detects it from the content."""
    if source == "auto":
        source = detect_weather_source(content)
        log.info(f"Weather source detected from the content: {source}")
    if source not in WEATHER_SOURCES:
        raise ValueError(f"Unknown weather source {source!r}; known: auto, {', '.join(WEATHER_SOURCES)}")
    return WEATHER_SOURCES[source](content, **kwargs).load()


def resolve_site(weather, latitude=None, longitude=None, altitude=None, max_distance_km=50.0, **kwargs):
    """Site coordinates: from the config's site section if given, else from the weather data.

    Gridded weather data (e.g. Open-Meteo, about 25 km grid) comes from a point near the site; the distance is
    logged, and a warning is given beyond `max_distance_km`, which likely means a wrong site or weather file.
    """
    site = {"latitude": latitude, "longitude": longitude, "altitude": altitude}
    for key in site:
        if site[key] is None:
            site[key] = getattr(weather, key)
    if site["latitude"] is None or site["longitude"] is None:
        raise ValueError(f"Site coordinates unknown: the weather source {weather.source!r} provides none; set site.latitude and site.longitude in the config")
    if site["altitude"] is None:
        site["altitude"] = 0.0
    if latitude is not None and longitude is not None and weather.latitude is not None and weather.longitude is not None:
        distance = distance_km(latitude, longitude, weather.latitude, weather.longitude)
        message = f"Site {latitude}°, {longitude}° from the config; the weather data is for {weather.latitude}°, {weather.longitude}°, {distance:.1f} km away"
        if distance > max_distance_km:
            log.warning(message + f" (more than {max_distance_km:g} km: wrong site or weather file?)")
        else:
            log.info(message)
    return site
