"""Python side of the browser front end, run in a Web Worker via Pyodide (see `web/worker.js`).

The page passes file contents and the config as strings and receives JSON strings (or bytes for downloads), so no
Python objects cross into JavaScript. Network requests (site search, weather download) are made by the page with
`fetch`; this module builds their URLs and parses the responses. The irradiation per sky patch is cached: changing
only the panel or orientation sections recomputes just the yield.
"""

import io
import json
import logging
import sys
import time
import zipfile

import numpy as np
from bokeh.embed import json_item

from pv_yield_estimator import __version__
from pv_yield_estimator.config import config_from_yaml, config_to_yaml, default_config, log_config, merge_configs
from pv_yield_estimator.core.export import export_files, monthly_rows
from pv_yield_estimator.core.irradiation import compute_patch_irradiation
from pv_yield_estimator.core.obstruction_methods import sky_obstruction_method
from pv_yield_estimator.core.panel_yield import YieldEstimator
from pv_yield_estimator.core.sky import ObstructedSky, SkyDiscretization
from pv_yield_estimator.core.weather import distance_km, load_weather, resolve_site
from pv_yield_estimator.core.weather_download import parse_site_name, parse_site_search, pvgis_tmy_url, site_name_url, site_search_url, weather_download_candidates
from pv_yield_estimator.file_format import to_json_text
from pv_yield_estimator.plotting.interactive import result_plots

log = logging.getLogger(__name__)


def versions():
    """Versions of the package and its main dependencies, as JSON; the page loads BokehJS matching `bokeh` exactly."""
    import bokeh
    import pandas
    import pvlib
    return json.dumps({"pv_yield_estimator": __version__, "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pandas.__version__, "pvlib": pvlib.__version__, "bokeh": bokeh.__version__})


def site_search(query):
    """URL of the site search for a place name or address (the page fetches it), as JSON string."""
    return json.dumps(site_search_url(query))


def site_search_results(text):
    """Places found by the site search (response text), as JSON list of {name, latitude, longitude}."""
    return json.dumps(parse_site_search(text))


def site_name(latitude, longitude):
    """URL of the reverse site search for coordinates (the page fetches it), as JSON string."""
    return json.dumps(site_name_url(latitude, longitude))


def site_name_result(text):
    """Address or place name from the reverse site search response (empty if none), as JSON string."""
    return json.dumps(parse_site_name(text))


def weather_downloads(latitude, longitude, weather_json="{}"):
    """Weather downloads the page can fetch for a site (CORS-enabled services), plus the PVGIS URL for a one-tap file download, as JSON."""
    weather = json.loads(weather_json)
    candidates = weather_download_candidates(latitude, longitude, in_browser=True, **{key: value for key, value in weather.items() if key != "download_service"})
    return json.dumps({"candidates": candidates, "pvgis_url": pvgis_tmy_url(latitude, longitude, browser_download=True, **weather)})


def config_yaml_from_json(config_json):
    """The config (JSON text of a partial config, merged over the defaults) as YAML text for saving."""
    return config_to_yaml(merge_configs(default_config(), json.loads(config_json)))


def config_json_from_yaml(text, filename=""):
    """A YAML config file's content as JSON text of the config dict (merged over the defaults)."""
    return json.dumps(config_from_yaml(text, source=filename or "<browser>"))


class BrowserSession:
    """State of the browser app: the loaded weather data and obstructed sky description, the cached irradiation per sky patch and the latest result."""

    def __init__(self):
        self.weather = None
        self.obstructed_sky = None
        self.irradiation = None
        self.irradiation_key = None
        self.last = None

    def load_weather(self, content, filename="", source="auto"):
        """Parse a weather file's content with the given weather source (auto: detected); returns a JSON summary for the page."""
        self.weather = load_weather(content, source=source)
        self.weather.metadata["input_file"] = filename
        self.irradiation = None
        annual = self.weather.hourly.sum() / 1000.0
        return json.dumps({"name": self.weather.name, "source": self.weather.source, "filename": filename, "latitude": self.weather.latitude, "longitude": self.weather.longitude, "altitude": self.weather.altitude,
                           "n_hours": len(self.weather.hourly), "annual_ghi_kwh_m2": float(annual["ghi"]), "annual_dhi_kwh_m2": float(annual["dhi"]), "annual_dni_kwh_m2": float(annual["dni"])})

    def set_obstructed_sky(self, obstructed_sky):
        """Use an obstructed sky description; returns a JSON summary for the page."""
        if self.irradiation is not None and not self.irradiation.sky.same_as(obstructed_sky.sky):
            self.irradiation = None
        self.obstructed_sky = obstructed_sky
        metadata = obstructed_sky.metadata
        return json.dumps({"n_patches": obstructed_sky.sky.n_patches, "n_obstructed": int(np.count_nonzero(obstructed_sky.obstructed)),
                           "obstructed_solid_angle_fraction": obstructed_sky.obstructed_solid_angle_fraction(), "sky_view_factor_horizontal": obstructed_sky.sky_view_factor(),
                           "method": metadata.get("method", ""), "created": metadata.get("created", ""), "input_file": metadata.get("input_file", "")})

    def load_obstructed_sky(self, text, filename=""):
        """Parse an obstructed sky description (JSON text, as written by the CLI or saved from this page); returns a JSON summary."""
        return self.set_obstructed_sky(ObstructedSky.from_dict(json.loads(text), source=filename or "<browser>"))

    def compute_obstruction(self, path, filename, config_json):
        """Obstructed sky description from a point cloud file in Pyodide's file system (written there by the worker), with the config's sky_obstruction, simulation and site sections; returns a JSON summary."""
        config = merge_configs(default_config(), json.loads(config_json))
        method = sky_obstruction_method(**config["sky_obstruction"])
        with open(path, encoding="utf-8") as point_cloud_file:
            data = method.read_input(point_cloud_file)
        sky = SkyDiscretization.from_node_count(**config["simulation"])
        site = {key: config["site"][key] for key in ("latitude", "longitude", "altitude", "name") if config["site"].get(key) is not None}
        return self.set_obstructed_sky(method.obstructed_sky(sky, data, metadata={"input_file": filename, **site}))

    def clear_obstructed_sky(self):
        """Forget the obstructed sky description: the next computation assumes no obstruction."""
        self.obstructed_sky = None

    def obstructed_sky_text(self):
        """The current obstructed sky description as JSON file content, for saving."""
        return to_json_text(self.obstructed_sky.to_dict())

    def compute(self, config_json):
        """Compute key figures, monthly values, orientation comparison and plots for a config (JSON text of a partial config, merged over the defaults); returns JSON.

        Without an obstructed sky description, the sky is free (no obstruction), on the discretization of the
        simulation section.
        """
        if self.weather is None:
            raise ValueError("Load or download weather data first")
        config = merge_configs(default_config(), json.loads(config_json))
        obstructed_sky = self.obstructed_sky or ObstructedSky.free(SkyDiscretization.from_node_count(**config["simulation"]))
        log_config(config)
        timings = {}
        site = resolve_site(self.weather, **config["site"])
        ### The irradiation depends on the weather, the sky discretization, the site and the simulation settings only.
        irradiation_key = json.dumps([site, config["simulation"]], sort_keys=True)
        if self.irradiation is None or irradiation_key != self.irradiation_key or not self.irradiation.sky.same_as(obstructed_sky.sky):
            start = time.perf_counter()
            self.irradiation = compute_patch_irradiation(self.weather, obstructed_sky.sky, **site, **config["simulation"])
            self.irradiation.metadata["input_file"] = self.weather.metadata.get("input_file", "")
            self.irradiation_key = irradiation_key
            timings["irradiation per sky patch"] = time.perf_counter() - start
        start = time.perf_counter()
        result = YieldEstimator(self.irradiation, obstructed_sky, **config).run()
        key_figures = result.key_figures()
        timings["yield"] = time.perf_counter() - start
        start = time.perf_counter()
        plots = {name: json_item(plot) for name, plot in result_plots(result, obstructed_sky, self.irradiation).items()}
        timings["plots"] = time.perf_counter() - start
        self.last = {"result": result, "config": config, "site": site, "obstructed_sky": obstructed_sky}
        weather_distance_km = distance_km(site["latitude"], site["longitude"], self.weather.latitude, self.weather.longitude) if self.weather.latitude is not None else None
        return json.dumps({"config": config, "site": site, "weather_distance_km": weather_distance_km, "key_figures": key_figures, "optimized_angles": result.optimized_angles,
                           "monthly_daily_average": monthly_rows(result), "orientation_comparison": result.orientation_comparison, "plots": plots, "timings": timings})

    def require_result(self):
        """The latest result, or an error asking to compute first."""
        if self.last is None:
            raise ValueError("Compute the results first")
        return self.last

    def export_zip(self):
        """The export files of the latest result (results.json, config.yaml, CSV tables) plus the inputs' JSON, as zip file bytes."""
        last = self.require_result()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, text in export_files(last["result"], last["config"], last["site"], self.irradiation, last["obstructed_sky"]).items():
                archive.writestr(name, text)
            archive.writestr("obstructed_sky.json", to_json_text(last["obstructed_sky"].to_dict()))
        return buffer.getvalue()

    def pdf_report(self):
        """PDF report of the latest result as bytes; needs matplotlib, which the worker loads on first use."""
        from pv_yield_estimator.plotting.report import pdf_report
        last = self.require_result()
        return pdf_report(last["result"], last["config"], last["site"], self.irradiation, last["obstructed_sky"])
