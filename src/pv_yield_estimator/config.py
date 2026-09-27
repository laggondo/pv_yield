"""Config layer shared by the front ends: the internal config dict, YAML import/export and overrides.

The config is a nested dict with one sub-dict per top-level section. Consumers receive sections splatted as keyword
arguments, so each entry and its default are documented in the signature of the consuming function. Only the front
ends assemble or change the config; this module works on strings and dicts, not files.
"""

import copy
import logging

import yaml

from pv_yield_estimator.file_format import FORMAT_VERSION_KEY, check_format_version

log = logging.getLogger(__name__)

CONFIG_FORMAT_VERSION = 1
CONFIG_SECTIONS = ("site", "weather", "sky_obstruction", "panel", "simulation", "output")


def default_config():
    """Return a config with all known sections, each empty; defaults live in the signatures of the consumers."""
    return {section: {} for section in CONFIG_SECTIONS}


def merge_configs(base, update):
    """Return a deep merge of two nested dicts; entries of `update` win, nested dicts are merged recursively."""
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def config_from_yaml(text, source="<string>"):
    """Parse a YAML config, check its format version and return it merged over the default config.

    The format version is only part of the stored file, not of the internal config dict. A missing version is
    accepted with a warning, so hand-written configs may omit it.
    """
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    check_format_version(data, CONFIG_FORMAT_VERSION, "Config", source, missing_ok=True)
    data = {key: value for key, value in data.items() if key != FORMAT_VERSION_KEY}
    for section, entries in data.items():
        if section not in CONFIG_SECTIONS:
            log.warning(f"Config from {source}: unknown section {section!r} (known: {', '.join(CONFIG_SECTIONS)})")
        if entries is None:
            data[section] = {}
        elif not isinstance(entries, dict):
            raise ValueError(f"Config from {source}: section {section!r} must be a mapping, got {entries!r}")
    return merge_configs(default_config(), data)


def config_to_yaml(config):
    """Export the config as block-style YAML, with the format version first."""
    return yaml.safe_dump({FORMAT_VERSION_KEY: CONFIG_FORMAT_VERSION, **config}, sort_keys=False, default_flow_style=False, allow_unicode=True)


def parse_overrides(items):
    """Parse `KEY=VALUE` strings into a dict; the value is parsed with YAML and the split is on the first `=` only."""
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must have the form KEY=VALUE, e.g. panel.tilt_deg=30")
    return dict([(lambda k, v: (k, yaml.safe_load(v)))(*item.split("=", 1)) for item in items])


def apply_overrides(config, overrides):
    """Return a copy of the config with overrides applied; dotted keys (`section.entry`) address nested entries."""
    config = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        *parent_keys, last_key = dotted_key.split(".")
        if (parent_keys or [last_key])[0] not in CONFIG_SECTIONS:
            log.warning(f"Override {dotted_key!r}: unknown section (known: {', '.join(CONFIG_SECTIONS)})")
        node = config
        for depth, key in enumerate(parent_keys):
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Override {dotted_key!r}: {'.'.join(parent_keys[:depth + 1])!r} is {node!r}, not a section")
        node[last_key] = value
    return config


def log_config(config):
    """Log the assembled config at info level as block-style YAML."""
    log.info("Config:\n" + config_to_yaml(config))
