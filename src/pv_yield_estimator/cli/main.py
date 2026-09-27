"""Entry point of the command-line front end.

Subcommands run the pipeline steps, each reading and writing intermediate files:
- `obstruction`: LiDAR point cloud → obstructed sky description (JSON)
- `irradiation`: weather file → irradiation per sky patch (JSON)
- `yield`: irradiation per sky patch + obstructed sky description → key figures (printed, optionally YAML/CSV)
- `run`: all steps in one go, writing the intermediate files into an output directory
Without a subcommand, the config is only assembled, logged and optionally exported.
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

from pv_yield_estimator import __version__
from pv_yield_estimator.config import apply_overrides, config_from_yaml, config_to_yaml, default_config, log_config, merge_configs, parse_overrides
from pv_yield_estimator.core.irradiation import PatchIrradiation, compute_patch_irradiation
from pv_yield_estimator.core.obstruction_methods import sky_obstruction_method
from pv_yield_estimator.core.panel_yield import YieldEstimator
from pv_yield_estimator.core.sky import ObstructedSky, SkyDiscretization
from pv_yield_estimator.core.weather import load_weather, resolve_site
from pv_yield_estimator.file_format import to_json_text
from pv_yield_estimator.logging_setup import setup_logging

log = logging.getLogger(__name__)


def add_common_arguments(parser, defaults=True):
    """Add the config and logging options; subcommands get them without defaults, so they only override if given."""
    suppress = {} if defaults else {"default": argparse.SUPPRESS}
    parser.add_argument("-c", "--config", action="append", type=Path, **({"default": []} | suppress), help="YAML config file; may be repeated, later files override earlier ones")
    parser.add_argument("-m", "--mod", nargs="+", action="extend", metavar="KEY=VALUE", **({"default": []} | suppress), help="override config entries, e.g. -m panel.tilt_deg=30 site.latitude=48.0; values are parsed as YAML")
    parser.add_argument("--export-config", type=Path, metavar="FILE", **({"default": None} | suppress), help="write the assembled config to this YAML file")
    parser.add_argument("-l", "--log-level", **({"default": "info"} | suppress), help="log level: debug, info, warning, error (default: info)")


def parse_arguments(argv=None):
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="pv-yield-estimator", description="Estimate photovoltaic yield.")
    add_common_arguments(parser)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    obstruction = subparsers.add_parser("obstruction", help="LiDAR point cloud -> obstructed sky description (JSON)")
    obstruction.add_argument("point_cloud", type=Path, help="point cloud file, e.g. Livox CSV")
    obstruction.add_argument("-o", "--output", type=Path, required=True, help="obstructed sky description to write (JSON)")

    irradiation = subparsers.add_parser("irradiation", help="weather file -> irradiation per sky patch (JSON)")
    irradiation.add_argument("weather", type=Path, help="weather file, e.g. TMY3 CSV")
    irradiation.add_argument("-o", "--output", type=Path, required=True, help="irradiation per sky patch to write (JSON)")
    irradiation.add_argument("--sky", type=Path, help="obstructed sky description whose sky discretization to use (default: from simulation.n_sky_nodes)")

    yield_parser = subparsers.add_parser("yield", help="irradiation per sky patch + obstructed sky description -> key figures")
    yield_parser.add_argument("irradiation", type=Path, help="irradiation per sky patch (JSON)")
    yield_parser.add_argument("obstructed_sky", type=Path, help="obstructed sky description (JSON)")
    add_result_arguments(yield_parser)

    run = subparsers.add_parser("run", help="all steps: point cloud + weather file -> key figures, intermediate files in a directory")
    run.add_argument("point_cloud", type=Path, help="point cloud file, e.g. Livox CSV")
    run.add_argument("weather", type=Path, help="weather file, e.g. TMY3 CSV")
    run.add_argument("-d", "--output-dir", type=Path, required=True, help="directory for obstructed_sky.json, patch_irradiation.json and key_figures.yaml")

    for subparser in subparsers.choices.values():
        add_common_arguments(subparser, defaults=False)
    return parser.parse_args(argv)


def add_result_arguments(parser):
    """Options for writing the results of the yield step."""
    parser.add_argument("-o", "--output", type=Path, help="write config, key figures and monthly values to this YAML file")
    parser.add_argument("--hourly-csv", type=Path, metavar="FILE", help="write the hourly radiation on the panel (Wh/m²) to this CSV file")


def assemble_config(config_paths=(), override_items=()):
    """Load the config files in order, merge them over the default config and apply the overrides."""
    config = default_config()
    for path in config_paths:
        config = merge_configs(config, config_from_yaml(path.read_text(encoding="utf-8"), source=str(path)))
    return apply_overrides(config, parse_overrides(override_items))


def write_json(path, data):
    """Write a file's dict as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json_text(data), encoding="utf-8")
    log.info(f"Wrote {path}")


def read_json(path):
    """Read a JSON file into a dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def run_obstruction(point_cloud_path, config):
    """Obstructed sky description from a point cloud file, with the configured method and sky discretization."""
    method = sky_obstruction_method(**config["sky_obstruction"])
    with open(point_cloud_path, encoding="utf-8") as point_cloud_file:
        data = method.read_input(point_cloud_file)
    sky = SkyDiscretization.from_node_count(**config["simulation"])
    site = {key: config["site"][key] for key in ("latitude", "longitude", "altitude", "name") if key in config["site"]}
    return method.obstructed_sky(sky, data, metadata={"input_file": point_cloud_path.name, **site})


def run_irradiation(weather_path, config, sky=None):
    """Irradiation per sky patch from a weather file; sky discretization from the config unless given."""
    weather = load_weather(weather_path.read_text(encoding="utf-8"), **config["weather"])
    site = resolve_site(weather, **config["site"])
    if sky is None:
        sky = SkyDiscretization.from_node_count(**config["simulation"])
    irradiation = compute_patch_irradiation(weather, sky, **site, **config["simulation"])
    irradiation.metadata["input_file"] = weather_path.name
    return irradiation


def run_yield(irradiation, obstructed_sky, config, output=None, hourly_csv=None):
    """Compute the results, log the key figures and monthly values, and write them if requested."""
    result = YieldEstimator(irradiation, obstructed_sky, **config).run()
    key_figures = result.key_figures()
    monthly = result.monthly_daily_average()
    log.info("Key figures (radiation in kWh/m², yield in kWh, specific yield in kWh/kWp):\n" + yaml.safe_dump({key: round(value, 4) for key, value in key_figures.items()}, sort_keys=False))
    summary = monthly[["total_unobstructed", "total_obstructed", "yield_unobstructed", "yield_obstructed"]].round(3)
    summary.columns = ["rad_unobs", "rad_obs", "yield_unobs", "yield_obs"]
    log.info(f"Average daily values per month: radiation on the panel (kWh/m²/d) and yield (kWh/d), unobstructed and obstructed:\n{summary.to_string()}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        content = {"config": config, "key_figures": key_figures, "monthly_daily_average": {column: {int(month): float(value) for month, value in values.items()} for column, values in monthly.items()}}
        output.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")
        log.info(f"Wrote {output}")
    if hourly_csv is not None:
        hourly_csv.parent.mkdir(parents=True, exist_ok=True)
        result.hourly.to_csv(hourly_csv, index_label="hour_start", float_format="%.3f")
        log.info(f"Wrote {hourly_csv}")
    return result


def main(argv=None):
    """Run the command-line front end."""
    args = parse_arguments(argv)
    setup_logging(args.log_level)
    log.info(f"pv_yield_estimator {__version__}")
    config = assemble_config(args.config, args.mod)
    log_config(config)
    if args.export_config is not None:
        args.export_config.write_text(config_to_yaml(config), encoding="utf-8")
        log.info(f"Wrote config to {args.export_config}")
    if args.command == "obstruction":
        write_json(args.output, run_obstruction(args.point_cloud, config).to_dict())
    elif args.command == "irradiation":
        sky = ObstructedSky.from_dict(read_json(args.sky), source=str(args.sky)).sky if args.sky else None
        write_json(args.output, run_irradiation(args.weather, config, sky).to_dict())
    elif args.command == "yield":
        irradiation = PatchIrradiation.from_dict(read_json(args.irradiation), source=str(args.irradiation))
        obstructed_sky = ObstructedSky.from_dict(read_json(args.obstructed_sky), source=str(args.obstructed_sky))
        run_yield(irradiation, obstructed_sky, config, args.output, args.hourly_csv)
    elif args.command == "run":
        obstructed_sky = run_obstruction(args.point_cloud, config)
        write_json(args.output_dir / "obstructed_sky.json", obstructed_sky.to_dict())
        irradiation = run_irradiation(args.weather, config, obstructed_sky.sky)
        write_json(args.output_dir / "patch_irradiation.json", irradiation.to_dict())
        run_yield(irradiation, obstructed_sky, config, args.output_dir / "key_figures.yaml", args.output_dir / "hourly_panel_radiation.csv")


if __name__ == "__main__":
    main()
