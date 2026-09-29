"""Python side of the browser front end, run in a Web Worker via Pyodide (see `web/worker.js`).

The page passes file contents and the config as strings and receives JSON strings, so no Python objects cross into
JavaScript. The irradiation per sky patch is cached: changing only the panel section recomputes just the yield.
"""

import json
import logging
import sys
import time

import numpy as np
from bokeh.embed import json_item

from pv_yield_estimator import __version__
from pv_yield_estimator.config import default_config, log_config, merge_configs
from pv_yield_estimator.core.irradiation import compute_patch_irradiation
from pv_yield_estimator.core.panel_yield import YieldEstimator
from pv_yield_estimator.core.sky import ObstructedSky
from pv_yield_estimator.core.weather import load_weather, resolve_site
from pv_yield_estimator.plotting.interactive import result_plots

log = logging.getLogger(__name__)


def versions():
    """Versions of the package and its main dependencies, as JSON; the page loads BokehJS matching `bokeh` exactly."""
    import bokeh
    import pandas
    import pvlib
    return json.dumps({"pv_yield_estimator": __version__, "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pandas.__version__, "pvlib": pvlib.__version__, "bokeh": bokeh.__version__})


class BrowserSession:
    """State of the browser app: the loaded weather data and obstructed sky description, and the cached irradiation per sky patch."""

    def __init__(self):
        self.weather = None
        self.obstructed_sky = None
        self.irradiation = None
        self.irradiation_key = None

    def load_weather(self, content, filename="", source="pvgis_tmy"):
        """Parse a weather file's content with the given weather source; returns a JSON summary for the page."""
        self.weather = load_weather(content, source=source)
        self.weather.metadata["input_file"] = filename
        self.irradiation = None
        annual = self.weather.hourly.sum() / 1000.0
        return json.dumps({"name": self.weather.name, "source": source, "filename": filename, "latitude": self.weather.latitude, "longitude": self.weather.longitude, "altitude": self.weather.altitude,
                           "n_hours": len(self.weather.hourly), "annual_ghi_kwh_m2": float(annual["ghi"]), "annual_dhi_kwh_m2": float(annual["dhi"]), "annual_dni_kwh_m2": float(annual["dni"])})

    def load_obstructed_sky(self, text, filename=""):
        """Parse an obstructed sky description (JSON text, as written by the CLI); returns a JSON summary for the page."""
        obstructed_sky = ObstructedSky.from_dict(json.loads(text), source=filename or "<browser>")
        if self.irradiation is not None and not self.irradiation.sky.same_as(obstructed_sky.sky):
            self.irradiation = None
        self.obstructed_sky = obstructed_sky
        metadata = obstructed_sky.metadata
        return json.dumps({"filename": filename, "n_patches": obstructed_sky.sky.n_patches, "n_obstructed": int(np.count_nonzero(obstructed_sky.obstructed)),
                           "obstructed_solid_angle_fraction": obstructed_sky.obstructed_solid_angle_fraction(), "sky_view_factor_horizontal": obstructed_sky.sky_view_factor(),
                           "method": metadata.get("method", ""), "created": metadata.get("created", ""), "input_file": metadata.get("input_file", "")})

    def compute(self, config_json):
        """Compute key figures, monthly values and plots for a config (JSON text of a partial config, merged over the defaults); returns JSON."""
        if self.weather is None or self.obstructed_sky is None:
            raise ValueError(f"Load a weather file and an obstructed sky description first (weather loaded: {self.weather is not None}, obstructed sky loaded: {self.obstructed_sky is not None})")
        config = merge_configs(default_config(), json.loads(config_json))
        log_config(config)
        timings = {}
        site = resolve_site(self.weather, **config["site"])
        ### The irradiation depends on the weather, the sky discretization, the site and the simulation settings only.
        irradiation_key = json.dumps([site, config["simulation"]], sort_keys=True)
        if self.irradiation is None or irradiation_key != self.irradiation_key:
            start = time.perf_counter()
            self.irradiation = compute_patch_irradiation(self.weather, self.obstructed_sky.sky, **site, **config["simulation"])
            self.irradiation_key = irradiation_key
            timings["irradiation per sky patch"] = time.perf_counter() - start
        start = time.perf_counter()
        result = YieldEstimator(self.irradiation, self.obstructed_sky, **config).run()
        key_figures = result.key_figures()
        monthly = result.monthly_daily_average()
        timings["yield"] = time.perf_counter() - start
        start = time.perf_counter()
        plots = {name: json_item(plot) for name, plot in result_plots(result, self.obstructed_sky, self.irradiation).items()}
        timings["plots"] = time.perf_counter() - start
        monthly_rows = [{"month": int(month), **{column: float(value) for column, value in row.items()}} for month, row in monthly.iterrows()]
        return json.dumps({"config": config, "site": site, "key_figures": key_figures, "monthly_daily_average": monthly_rows, "plots": plots, "timings": timings})
