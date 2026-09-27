"""Format versions and JSON text of stored files (configs, obstructed sky description, irradiation per sky patch)."""

import json
import logging

log = logging.getLogger(__name__)

FORMAT_VERSION_KEY = "format_version"


def check_format_version(data, current_version, kind, source="<unknown>", missing_ok=False):
    """Check the format version of loaded file content and return it.

    Raises if the version is missing (unless `missing_ok`, then the current version is assumed), not an integer,
    or newer than `current_version`. Older versions are returned, so the caller can convert them.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{kind} from {source} must be a mapping at the top level, got {type(data).__name__}")
    if FORMAT_VERSION_KEY not in data:
        if not missing_ok:
            raise ValueError(f"{kind} from {source} has no '{FORMAT_VERSION_KEY}' entry")
        log.warning(f"{kind} from {source} has no '{FORMAT_VERSION_KEY}' entry; assuming the current version {current_version}")
        return current_version
    version = data[FORMAT_VERSION_KEY]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{kind} from {source}: '{FORMAT_VERSION_KEY}' must be a positive integer, got {version!r}")
    if version > current_version:
        raise ValueError(f"{kind} from {source} has format version {version}, but this program reads up to version {current_version}; update the program")
    return version


def to_json_text(data):
    """Serialize a file's top-level dict as JSON that stays readable: one line per top-level entry and per element of top-level lists."""
    lines = []
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], (list, dict)):
            text = "[\n" + ",\n".join("    " + json.dumps(element) for element in value) + "\n  ]"
        else:
            text = json.dumps(value)
        lines.append(f"  {json.dumps(key)}: {text}")
    return "{\n" + ",\n".join(lines) + "\n}\n"
