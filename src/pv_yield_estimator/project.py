"""Project files: all files of a project (config, weather, obstruction, photos, point cloud, results) in one zip (#37).

The zip has fixed file names, so loading needs no questions; `project.yaml` lists the contents, with the format
version, the creation date and the program version. Every file is optional: saving adds a file only if it is
available, and loading uses the files that are there. Front-end independent: the browser app builds and reads the
zip in its worker, and the CLI could read and write the same zip.
"""

import datetime
import logging
import zipfile
from pathlib import Path, PurePosixPath

import yaml

from pv_yield_estimator import __version__
from pv_yield_estimator.file_format import FORMAT_VERSION_KEY, check_format_version

log = logging.getLogger(__name__)

PROJECT_FORMAT_VERSION = 1
MANIFEST_NAME = "project.yaml"
CONFIG_NAME = "config.yaml"
OBSTRUCTED_SKY_NAME = "obstructed_sky.json"
PHOTOS_NAME = "photos.json"
PHOTOS_DIRECTORY = "photos"
IRRADIATION_NAME = "irradiation.json"
REPORT_NAME = "report.pdf"
RESULTS_DIRECTORY = "results"
### Stems of files kept in their own format, with the original file's suffix (e.g. weather.csv or weather.json).
WEATHER_STEM = "weather"
POINT_CLOUD_STEM = "point_cloud"
### Already compressed formats are stored as they are.
STORED_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf", ".zip"}


def project_file_name(stem, original_filename, default_suffix=".csv"):
    """File name in the project for a file kept in its own format: the stem plus the original file's suffix (lower case), e.g. weather.csv."""
    suffix = PurePosixPath(original_filename or "").suffix.lower()
    return stem + (suffix if suffix else default_suffix)


def write_project_zip(target, contents, **details):
    """Write a project zip to `target` (path or binary file object): `contents` maps file names in the zip to text, bytes or a path of a file to copy.

    `project.yaml` is written first; it lists the contents and holds the `details` (e.g. the original file names).
    Returns the list of file names written besides `project.yaml`.
    """
    names = sorted(contents)
    manifest = {FORMAT_VERSION_KEY: PROJECT_FORMAT_VERSION, "kind": "project", "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "program_version": __version__, "contents": names, **details}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False, allow_unicode=True))
        for name in names:
            content = contents[name]
            compress_type = zipfile.ZIP_STORED if PurePosixPath(name).suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            if isinstance(content, Path):
                archive.write(content, name, compress_type=compress_type)
            else:
                archive.writestr(name, content, compress_type=compress_type)
    log.info(f"Project with {len(names)} files: {', '.join(names)}")
    return names


def read_project_manifest(archive, source="<project>"):
    """Read and check `project.yaml` of an open project zip; returns it. Files listed but missing are logged, not an error."""
    names = set(archive.namelist())
    if MANIFEST_NAME not in names:
        raise ValueError(f"{source} is not a project: it has no {MANIFEST_NAME} (found: {', '.join(sorted(names)[:10]) or 'nothing'})")
    manifest = yaml.safe_load(archive.read(MANIFEST_NAME).decode("utf-8"))
    check_format_version(manifest, PROJECT_FORMAT_VERSION, "Project", source)
    if manifest.get("kind") != "project":
        raise ValueError(f"{source}: {MANIFEST_NAME} has kind {manifest.get('kind')!r}, expected 'project'")
    missing = [name for name in manifest.get("contents", []) if name not in names]
    if missing:
        log.warning(f"{source}: files listed in {MANIFEST_NAME} but missing: {', '.join(missing)}")
    return manifest


def find_project_file(archive, stem):
    """Name of the top-level file with the given stem (any suffix, e.g. weather.csv or weather.json) in an open project zip, or None."""
    matches = sorted(name for name in archive.namelist() if "/" not in name and PurePosixPath(name).stem == stem)
    return matches[0] if matches else None
