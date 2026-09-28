"""Smoke tests: the package imports, logging is set up and the CLI runs."""

import logging

import pytest
from rich.logging import RichHandler

import pv_yield_estimator
from pv_yield_estimator.cli.main import main
from pv_yield_estimator.logging_setup import setup_logging


def test_package_has_version():
    """The package imports and exposes a version string."""
    assert isinstance(pv_yield_estimator.__version__, str)


def test_setup_logging_installs_single_rich_handler():
    """setup_logging replaces the root handlers with one rich handler at the requested level."""
    setup_logging("debug")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1 and isinstance(root.handlers[0], RichHandler)
    setup_logging()
    assert root.level == logging.INFO


def test_setup_logging_rejects_unknown_level():
    """An unknown level name raises an error naming the value."""
    with pytest.raises(ValueError, match="verbose"):
        setup_logging("verbose")


def test_cli_runs(capsys):
    """The CLI runs and logs its version through the rich handler."""
    main([])
    assert pv_yield_estimator.__version__ in capsys.readouterr().out
