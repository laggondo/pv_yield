"""Export of the results: JSON with the config used, config as YAML, tables as CSV. Returns file contents; the front ends write or download them.

Every file carries a format version: the JSON and YAML files as `format_version` entry, the CSV files in a first
comment line (read them e.g. with `pandas.read_csv(path, comment="#")`).
"""

import datetime

import numpy as np
import pandas as pd

from pv_yield_estimator import __version__
from pv_yield_estimator.config import config_to_yaml
from pv_yield_estimator.core.orientation import SHADING_CASES
from pv_yield_estimator.file_format import FORMAT_VERSION_KEY, to_json_text

RESULTS_FORMAT_VERSION = 1


def monthly_rows(result):
    """Average daily values per month as a list of dicts (month 1–12, radiation in kWh/m²/d, yield in kWh/d)."""
    return [{"month": int(month), **{column: float(value) for column, value in row.items()}} for month, row in result.monthly_daily_average().iterrows()]


def results_to_dict(result, config, site, irradiation=None, obstructed_sky=None):
    """Content of the results JSON file: config and site used, input metadata, key figures, monthly values and the orientation comparison."""
    grid = result.orientation_grid
    return {
        FORMAT_VERSION_KEY: RESULTS_FORMAT_VERSION, "kind": "results", "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "program_version": __version__,
        "config": config, "site": site,
        "inputs": {"weather": dict(irradiation.metadata) if irradiation is not None else {}, "obstructed_sky": dict(obstructed_sky.metadata) if obstructed_sky is not None else {}},
        "optimized_angles": result.optimized_angles, "key_figures": result.key_figures(), "monthly_daily_average": monthly_rows(result),
        "orientation": {"comparison": result.orientation_comparison, **({f"best_{case}": grid.best(case) for case in SHADING_CASES} if grid is not None else {})},
    }


def orientation_grid_table(grid):
    """The orientation grid in long form: one row per tilt and azimuth with the annual radiation (kWh/m²) unobstructed and obstructed."""
    tilts, azimuths = (values.ravel() for values in np.meshgrid(grid.tilts_deg, grid.azimuths_deg, indexing="ij"))
    return pd.DataFrame({"tilt_deg": tilts, "azimuth_deg": azimuths, **{f"radiation_{case}_kwh_m2": grid.radiation[case].ravel() for case in SHADING_CASES}})


def results_tables(result):
    """Result tables by name: hourly (Wh/m²), daily (kWh/m²), monthly (kWh/m²/d, yield kWh/d), orientation comparison and grid."""
    tables = {"hourly": result.hourly.rename_axis("hour_start"), "daily": result.daily().rename_axis("day"), "monthly": result.monthly_daily_average()}
    tables["daily"].index = tables["daily"].index.strftime("%Y-%m-%d")
    if result.orientation_comparison:
        tables["orientation_comparison"] = pd.DataFrame(result.orientation_comparison)
    if result.orientation_grid is not None:
        tables["orientation_grid"] = orientation_grid_table(result.orientation_grid)
    return tables


def csv_text(table, description):
    """CSV text of a table, with a first comment line naming the content and the format version."""
    index = not isinstance(table.index, pd.RangeIndex)
    return f"# pv_yield_estimator {description}, {FORMAT_VERSION_KEY} {RESULTS_FORMAT_VERSION}\n" + table.to_csv(index=index, float_format="%.4f", lineterminator="\n")


CSV_DESCRIPTIONS = {"hourly": "hourly radiation on the panel (Wh/m²)", "daily": "radiation on the panel per day (kWh/m²)", "monthly": "average daily values per month (radiation kWh/m²/d, yield kWh/d)",
                    "orientation_comparison": "orientation comparison (radiation kWh/m², yield kWh, specific yield kWh/kWp)", "orientation_grid": "annual radiation on the panel over tilt and azimuth (kWh/m²)"}


def export_files(result, config, site, irradiation=None, obstructed_sky=None):
    """All export files as {file name: text}: results.json, config.yaml and one CSV file per result table."""
    files = {"results.json": to_json_text(results_to_dict(result, config, site, irradiation, obstructed_sky)), "config.yaml": config_to_yaml(config)}
    files |= {f"{name}.csv": csv_text(table, CSV_DESCRIPTIONS[name]) for name, table in results_tables(result).items()}
    return files
