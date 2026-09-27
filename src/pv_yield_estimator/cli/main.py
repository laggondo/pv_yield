"""Entry point of the command-line front end."""

import argparse
import logging

from pv_yield_estimator import __version__
from pv_yield_estimator.logging_setup import setup_logging

log = logging.getLogger(__name__)


def parse_arguments(argv=None):
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="pv-yield-estimator", description="Estimate photovoltaic yield.")
    parser.add_argument("-l", "--log-level", default="info", help="log level: debug, info, warning, error (default: info)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the command-line front end."""
    args = parse_arguments(argv)
    setup_logging(args.log_level)
    log.info(f"pv_yield_estimator {__version__}")


if __name__ == "__main__":
    main()
