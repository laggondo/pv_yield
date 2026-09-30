"""Automatic weather download: request URLs of the free weather services, the choice of service for a site, and the site search.

The core does no network access: it builds the request URLs and names the weather source that parses the response;
the front ends fetch (the CLI with urllib, the browser with `fetch`) and keep the response as a file, so later steps
can run offline. Services:
- `pvgis_tmy`: PVGIS typical meteorological year (satellite data, preferred where available). The PVGIS API sends no
  CORS headers, so a browser page cannot fetch it; the browser offers the URL as a link for a manual download instead.
- `open_meteo`: Open-Meteo archive of real hourly years (reanalysis, global, CORS-enabled), averaged over several
  years into a typical year.
The site search (geocoding: place name or address → coordinates) uses Nominatim (OpenStreetMap), which is CORS-enabled.
"""

import datetime
import json
import logging
from urllib.parse import urlencode

log = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/v5_3/tmy"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
### Identifies the program towards the web services, as Nominatim's usage policy asks (the CLI sends it as User-Agent).
USER_AGENT = "pv_yield_estimator (https://github.com/laggondo/pv_yield)"


def open_meteo_years(n_years=10, end_year=None, today=None, **kwargs):
    """First and last year of the Open-Meteo download: `n_years` complete years up to `end_year` (default: last year)."""
    if end_year is None:
        end_year = (today or datetime.date.today()).year - 1
    if n_years < 1:
        raise ValueError(f"weather.n_years must be at least 1, got {n_years}")
    return end_year - n_years + 1, end_year


def open_meteo_url(latitude, longitude, n_years=10, end_year=None, open_meteo_model=None, **kwargs):
    """Request URL of the Open-Meteo archive API for hourly global, diffuse and direct normal radiation over whole years, in UTC.

    `open_meteo_model` selects the data set (e.g. `era5`, `era5_land`); by default Open-Meteo chooses (`best_match`).
    """
    first_year, last_year = open_meteo_years(n_years, end_year)
    parameters = {"latitude": f"{latitude:.4f}", "longitude": f"{longitude:.4f}", "start_date": f"{first_year}-01-01", "end_date": f"{last_year}-12-31",
                  "hourly": "shortwave_radiation,diffuse_radiation,direct_normal_irradiance", "timezone": "GMT"}
    if open_meteo_model:
        parameters["models"] = open_meteo_model
    return f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(parameters)}"


def pvgis_tmy_url(latitude, longitude, pvgis_use_horizon=True, **kwargs):
    """Request URL of the PVGIS typical meteorological year as CSV (the format of `data/Freiburg-pvgis-tmy.csv`).

    PVGIS shades the direct radiation by the terrain horizon from its elevation model unless `pvgis_use_horizon` is
    false; this covers distant mountains that a LiDAR scan does not reach.
    """
    parameters = {"lat": f"{latitude:.4f}", "lon": f"{longitude:.4f}", "outputformat": "csv", "usehorizon": int(bool(pvgis_use_horizon))}
    return f"{PVGIS_TMY_URL}?{urlencode(parameters)}"


### Download services by name: request URL builder, weather source that reads the response, file extension of the
### response, and whether a browser page may fetch it (CORS).
DOWNLOAD_SERVICES = {
    "pvgis_tmy": {"url": pvgis_tmy_url, "weather_source": "pvgis_tmy", "extension": "csv", "browser_fetch": False, "description": "PVGIS typical meteorological year (satellite data)"},
    "open_meteo": {"url": open_meteo_url, "weather_source": "open_meteo", "extension": "json", "browser_fetch": True, "description": "Open-Meteo archive (reanalysis), mean of several real years"},
}
### Order of preference for `download_service: auto`: PVGIS's satellite-based typical year first; it has no data over
### the sea and cannot be fetched from a browser page, so Open-Meteo (global) follows.
AUTO_SERVICE_ORDER = ("pvgis_tmy", "open_meteo")


def weather_download_candidates(latitude, longitude, download_service="auto", in_browser=False, **kwargs):
    """Planned downloads for a site, in order of preference; the front end fetches the first that succeeds.

    Each entry is a dict with service, description, url, weather_source (for `load_weather`), filename (for the
    cached file) and browser_fetch. `download_service` is a service name or `auto` (see `AUTO_SERVICE_ORDER`); in the
    browser, services without CORS are left out. Further weather section entries (e.g. n_years) reach the URL builders.
    """
    if latitude is None or longitude is None:
        raise ValueError(f"The weather download needs the site's coordinates; set site.latitude and site.longitude or site.query (got latitude {latitude!r}, longitude {longitude!r})")
    if download_service == "auto":
        names = [name for name in AUTO_SERVICE_ORDER if DOWNLOAD_SERVICES[name]["browser_fetch"] or not in_browser]
    elif download_service in DOWNLOAD_SERVICES:
        names = [download_service]
        if in_browser and not DOWNLOAD_SERVICES[download_service]["browser_fetch"]:
            raise ValueError(f"The weather service {download_service!r} cannot be fetched from a browser page (no CORS); download its file manually, or use {', '.join(name for name, service in DOWNLOAD_SERVICES.items() if service['browser_fetch'])}")
    else:
        raise ValueError(f"Unknown weather download service {download_service!r}; known: auto, {', '.join(DOWNLOAD_SERVICES)}")
    candidates = []
    for name in names:
        service = DOWNLOAD_SERVICES[name]
        suffix = "_{}-{}".format(*open_meteo_years(**kwargs)) if name == "open_meteo" else ""
        candidates.append({"service": name, "description": service["description"], "url": service["url"](latitude, longitude, **kwargs), "weather_source": service["weather_source"],
                           "filename": f"weather_{name}_{latitude:.3f}_{longitude:.3f}{suffix}.{service['extension']}", "browser_fetch": service["browser_fetch"]})
    return candidates


def site_search_url(query, limit=5):
    """Request URL of the Nominatim (OpenStreetMap) search for a place name or address."""
    if not str(query).strip():
        raise ValueError("The site search needs a place name or address, got an empty text")
    return f"{NOMINATIM_SEARCH_URL}?{urlencode({'q': query, 'format': 'jsonv2', 'limit': limit})}"


def parse_site_search(text):
    """Places found by the site search, best match first: list of dicts with name, latitude and longitude."""
    return [{"name": place["display_name"], "latitude": float(place["lat"]), "longitude": float(place["lon"])} for place in json.loads(text)]
