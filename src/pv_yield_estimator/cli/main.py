"""Entry point of the command-line front end."""

import argparse
import logging
from pathlib import Path

from pv_yield_estimator import __version__
from pv_yield_estimator.config import apply_overrides, config_from_yaml, config_to_yaml, default_config, log_config, merge_configs, parse_overrides
from pv_yield_estimator.logging_setup import setup_logging

log = logging.getLogger(__name__)


def parse_arguments(argv=None):
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="pv-yield-estimator", description="Estimate photovoltaic yield.")
    parser.add_argument("-c", "--config", action="append", default=[], type=Path, help="YAML config file; may be repeated, later files override earlier ones")
    parser.add_argument("-s", "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE", help="override config entries, e.g. -s panel.tilt_deg=30 site.latitude=48.0; values are parsed as YAML")
    parser.add_argument("--export-config", type=Path, metavar="FILE", help="write the assembled config to this YAML file")
    parser.add_argument("-l", "--log-level", default="info", help="log level: debug, info, warning, error (default: info)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def assemble_config(config_paths=(), override_items=()):
    """Load the config files in order, merge them over the default config and apply the overrides."""
    config = default_config()
    for path in config_paths:
        config = merge_configs(config, config_from_yaml(path.read_text(encoding="utf-8"), source=str(path)))
    return apply_overrides(config, parse_overrides(override_items))


def main(argv=None):
    """Run the command-line front end."""
    args = parse_arguments(argv)
    setup_logging(args.log_level)
    log.info(f"pv_yield_estimator {__version__}")
    config = assemble_config(args.config, args.set)
    log_config(config)
    if args.export_config is not None:
        args.export_config.write_text(config_to_yaml(config), encoding="utf-8")
        log.info(f"Wrote config to {args.export_config}")


if __name__ == "__main__":
    main()
