"""Tests of config loading, overrides, export and format version checks."""

import pytest
import yaml

from pv_yield_estimator.cli.main import assemble_config, main
from pv_yield_estimator.config import CONFIG_FORMAT_VERSION, CONFIG_SECTIONS, apply_overrides, config_from_yaml, config_to_yaml, default_config, merge_configs, parse_overrides
from pv_yield_estimator.file_format import check_format_version

EXAMPLE_CONFIG_YAML = """
format_version: 1
site:
  latitude: 47.99
  longitude: 7.84
panel:
  tilt_deg: 30
  azimuth_deg: 180
"""


def test_load_merges_over_default_and_drops_format_version():
    """Loading keeps all sections, fills missing ones and keeps the version out of the internal dict."""
    config = config_from_yaml(EXAMPLE_CONFIG_YAML)
    assert list(config) == list(CONFIG_SECTIONS)
    assert config["site"] == {"latitude": 47.99, "longitude": 7.84}
    assert config["weather"] == {}
    assert "format_version" not in config


def test_export_round_trip():
    """Export to YAML and load again gives the same config; the export is block style with the version first."""
    config = config_from_yaml(EXAMPLE_CONFIG_YAML)
    exported = config_to_yaml(config)
    assert exported.startswith(f"format_version: {CONFIG_FORMAT_VERSION}\n")
    assert "{latitude" not in exported
    assert config_from_yaml(exported) == config


def test_parse_overrides_yaml_values_and_first_equals_sign():
    """Values are parsed as YAML, and only the first `=` splits key from value."""
    overrides = parse_overrides(["panel.tilt_deg=30", "site.name=a=b", "simulation.use_sub_steps=true", "output.formats=[csv, json]"])
    assert overrides == {"panel.tilt_deg": 30, "site.name": "a=b", "simulation.use_sub_steps": True, "output.formats": ["csv", "json"]}


def test_parse_overrides_rejects_missing_equals_sign():
    """An override without `=` raises an error naming it."""
    with pytest.raises(ValueError, match="panel.tilt_deg"):
        parse_overrides(["panel.tilt_deg"])


def test_apply_overrides_nested_and_new_entries():
    """Dotted keys change nested entries, create missing ones and leave the input unchanged."""
    config = config_from_yaml(EXAMPLE_CONFIG_YAML)
    changed = apply_overrides(config, {"panel.tilt_deg": 45, "sky_obstruction.lidar.min_points": 3})
    assert changed["panel"] == {"tilt_deg": 45, "azimuth_deg": 180}
    assert changed["sky_obstruction"] == {"lidar": {"min_points": 3}}
    assert config["panel"]["tilt_deg"] == 30


def test_apply_overrides_rejects_entry_used_as_section():
    """Overriding below a non-dict entry raises an error naming the key."""
    with pytest.raises(ValueError, match="panel.tilt_deg"):
        apply_overrides(default_config() | {"panel": {"tilt_deg": 30}}, {"panel.tilt_deg.x": 1})


def test_merge_configs_is_deep():
    """Nested dicts are merged, other values replaced."""
    assert merge_configs({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"c": 3}, "d": [1]}) == {"a": {"b": 1, "c": 3}, "d": [1]}


def test_missing_version_accepted_for_configs(caplog):
    """Hand-written configs may omit the format version; a warning is logged."""
    config = config_from_yaml("panel:\n  tilt_deg: 20\n", source="hand.yaml")
    assert config["panel"] == {"tilt_deg": 20}
    assert "hand.yaml" in caplog.text


def test_empty_config_and_empty_section():
    """An empty file and an empty section both give empty sections."""
    assert config_from_yaml("") == default_config()
    assert config_from_yaml("format_version: 1\nsite:\n")["site"] == {}


def test_newer_version_rejected():
    """A file from a newer program version raises an error naming the source."""
    with pytest.raises(ValueError, match="new.yaml"):
        config_from_yaml(f"format_version: {CONFIG_FORMAT_VERSION + 1}\n", source="new.yaml")


def test_check_format_version_cases():
    """Missing (unless allowed), non-integer and non-positive versions are rejected; older ones are returned."""
    with pytest.raises(ValueError, match="no 'format_version'"):
        check_format_version({}, 1, "Sky description")
    for bad_version in ["1", 1.0, True, 0]:
        with pytest.raises(ValueError, match="positive integer"):
            check_format_version({"format_version": bad_version}, 1, "Sky description")
    assert check_format_version({"format_version": 1}, 2, "Sky description") == 1


def test_section_must_be_mapping():
    """A section that is not a mapping raises an error naming it."""
    with pytest.raises(ValueError, match="'panel'"):
        config_from_yaml("panel: 30\n")


def test_unknown_section_warns(caplog):
    """An unknown top-level section is kept but warned about (likely a typo)."""
    config = config_from_yaml("pannel:\n  tilt_deg: 20\n")
    assert config["pannel"] == {"tilt_deg": 20}
    assert "pannel" in caplog.text
    caplog.clear()
    apply_overrides(config, {"pannel.tilt_deg": 25})
    assert "unknown section" in caplog.text


def test_cli_assembles_and_exports(tmp_path):
    """The CLI merges config files in order, applies overrides and exports the result."""
    first, second, exported = tmp_path / "first.yaml", tmp_path / "second.yaml", tmp_path / "out.yaml"
    first.write_text(EXAMPLE_CONFIG_YAML)
    second.write_text("panel:\n  tilt_deg: 35\n")
    main(["-c", str(first), "-c", str(second), "-m", "panel.azimuth_deg=200", "site.latitude=48.0", "--export-config", str(exported)])
    config = config_from_yaml(exported.read_text())
    assert config == assemble_config([first, second], ["panel.azimuth_deg=200", "site.latitude=48.0"])
    assert config["panel"] == {"tilt_deg": 35, "azimuth_deg": 200}
    assert config["site"] == {"latitude": 48.0, "longitude": 7.84}
    assert yaml.safe_load(exported.read_text())["format_version"] == CONFIG_FORMAT_VERSION
