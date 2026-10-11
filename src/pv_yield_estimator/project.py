"""Project files: all files of a project (config, weather, obstruction, photos, AR recording, point cloud, results) in one zip (#37).

The zip has fixed file names, so loading needs no questions; `project.yaml` lists the contents, with the format
version, the creation date and the program version. Every file is optional: saving adds a file only if it is
available, and loading uses the files that are there. Front-end independent: the browser app builds and reads the
zip in its worker; the CLI reads and changes it with `ProjectFiles`, also as a directory with the same layout (e.g. an
unpacked zip).
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
### AR video recording (#47): the metadata, with the frames as recording/frame_<n>.jpg.
RECORDING_NAME = "recording.json"
### Point cloud reconstructed from the AR recording (#48).
RECONSTRUCTED_POINTS_NAME = "reconstructed_points.csv"
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


def project_manifest(names, **details):
    """Content of `project.yaml`: format version, kind, creation date, program version, the list of files and the `details` (e.g. the original file names)."""
    return {FORMAT_VERSION_KEY: PROJECT_FORMAT_VERSION, "kind": "project", "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "program_version": __version__, "contents": sorted(names), **details}


def manifest_yaml(manifest):
    """`project.yaml` as block-style YAML text."""
    return yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False, allow_unicode=True)


def write_project_zip(target, contents, **details):
    """Write a project zip to `target` (path or binary file object): `contents` maps file names in the zip to text, bytes or a path of a file to copy.

    `project.yaml` is written first; it lists the contents and holds the `details` (e.g. the original file names).
    Returns the list of file names written besides `project.yaml`.
    """
    names = sorted(contents)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, manifest_yaml(project_manifest(names, **details)))
        for name in names:
            content = contents[name]
            compress_type = zipfile.ZIP_STORED if PurePosixPath(name).suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            if isinstance(content, Path):
                archive.write(content, name, compress_type=compress_type)
            else:
                archive.writestr(name, content, compress_type=compress_type)
    log.info(f"Project with {len(names)} files: {', '.join(names) if len(names) <= 20 else ', '.join(names[:20]) + ', ...'}")
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


class ProjectFiles:
    """The files of a project for the CLI: a project zip or a directory with the same layout; changes are collected and written by `save`.

    A path that doesn't exist yet starts an empty project, a zip if it ends in `.zip`, else a directory. A zip is
    written anew on `save` (zip files can't replace entries), with the unchanged files copied over; a directory gets
    the changed files and a new `project.yaml`. The details in `project.yaml` (e.g. original file names) are kept and
    updated with `details`.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.is_zip = self.path.is_file() or (not self.path.exists() and self.path.suffix.lower() == ".zip")
        self.changes = {}
        self.removed = set()
        self.details = {}
        self.existing = set()
        if self.path.is_file():
            with self.open_zip() as archive:
                self.details = {key: value for key, value in read_project_manifest(archive, str(self.path)).items() if key not in (FORMAT_VERSION_KEY, "kind", "created", "program_version", "contents")}
                self.existing = {name for name in archive.namelist() if not name.endswith("/")} - {MANIFEST_NAME}
        elif self.path.is_dir():
            if (self.path / MANIFEST_NAME).is_file():
                manifest = yaml.safe_load((self.path / MANIFEST_NAME).read_text(encoding="utf-8")) or {}
                self.details = {key: value for key, value in manifest.items() if key not in (FORMAT_VERSION_KEY, "kind", "created", "program_version", "contents")}
            self.existing = {file.relative_to(self.path).as_posix() for file in self.path.rglob("*") if file.is_file()} - {MANIFEST_NAME}
        log.info(f"Project {self.path}: {len(self.existing)} files" if self.existing else f"Project {self.path}: new, empty")

    def open_zip(self):
        """The project zip, opened for reading."""
        try:
            return zipfile.ZipFile(self.path)
        except zipfile.BadZipFile as error:
            raise ValueError(f"{self.path} is not a zip file ({error}); give a project zip or a project directory") from error

    def names(self):
        """Names of the project's files (after the pending changes), without `project.yaml`."""
        return (self.existing - self.removed) | set(self.changes)

    def has(self, name):
        """Whether the project has the file."""
        return name in self.names()

    def find(self, stem):
        """Name of the top-level file with the given stem (any suffix, e.g. weather.csv), or None."""
        matches = sorted(name for name in self.names() if "/" not in name and PurePosixPath(name).stem == stem)
        return matches[0] if matches else None

    def read_bytes(self, name):
        """Content of a project file as bytes."""
        if name in self.changes:
            content = self.changes[name]
            return content.read_bytes() if isinstance(content, Path) else content.encode("utf-8") if isinstance(content, str) else content
        if name not in self.names():
            raise FileNotFoundError(f"Project {self.path} has no file {name!r} (it has: {', '.join(sorted(self.names())[:15]) or 'nothing'})")
        if self.is_zip:
            with self.open_zip() as archive:
                return archive.read(name)
        return (self.path / name).read_bytes()

    def read_text(self, name):
        """Content of a project file as text."""
        return self.read_bytes(name).decode("utf-8")

    def write(self, name, content):
        """Add or replace a file (text, bytes or a path of a file to copy); written by `save`."""
        self.changes[name] = content
        self.removed.discard(name)

    def write_stem_file(self, stem, original_filename, content, **details):
        """Add or replace a file kept in its own format (e.g. weather.csv), removing one of another suffix; its original file name and the `details` go into `project.yaml`. Returns the name in the project."""
        name = project_file_name(stem, original_filename)
        for other in [other for other in self.names() if "/" not in other and PurePosixPath(other).stem == stem and other != name]:
            self.remove(other)
        self.write(name, content)
        self.details[stem] = {"file": name, "original_filename": original_filename, **details}
        return name

    def remove(self, name):
        """Remove a file; done by `save`."""
        self.changes.pop(name, None)
        self.removed.add(name)

    def save(self):
        """Write the changes: a new zip (unchanged files copied), or the changed files and `project.yaml` into the directory."""
        if not self.changes and not self.removed:
            return
        names = self.names()
        if self.is_zip:
            contents = {name: self.changes.get(name) for name in names}
            temporary = self.path.with_name(self.path.name + ".partial")
            if self.path.is_file():
                with self.open_zip() as archive:
                    contents = {name: content if content is not None else archive.read(name) for name, content in contents.items()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_project_zip(temporary, contents, **self.details)
            temporary.replace(self.path)
        else:
            for name, content in self.changes.items():
                target = self.path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, Path):
                    target.write_bytes(content.read_bytes())
                else:
                    target.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
            for name in self.removed & self.existing:
                (self.path / name).unlink(missing_ok=True)
            (self.path / MANIFEST_NAME).write_text(manifest_yaml(project_manifest(names, **self.details)), encoding="utf-8")
        log.info(f"Wrote project {self.path}: changed {', '.join(sorted(self.changes)[:20]) or 'nothing'}" + (f", removed {', '.join(sorted(self.removed & self.existing))}" if self.removed & self.existing else ""))
        self.existing, self.changes, self.removed = names, {}, set()
