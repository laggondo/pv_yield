"""Tests of the project zip (#37): writing, reading the manifest, finding files kept in their own format."""

import io
import logging
import zipfile

import pytest
import yaml

from pv_yield_estimator.project import PROJECT_FORMAT_VERSION, find_project_file, project_file_name, read_project_manifest, write_project_zip


def test_project_zip_round_trip(tmp_path):
    """Text, bytes and files are written with project.yaml listing them; JPEGs are stored uncompressed; the manifest is read back."""
    point_cloud = tmp_path / "scan.csv"
    point_cloud.write_text("x,y,z\n1,2,3\n")
    buffer = io.BytesIO()
    names = write_project_zip(buffer, {"config.yaml": "panel: {}\n", "photos/photo_1.jpg": b"\xff\xd8jpeg", "point_cloud.csv": point_cloud}, weather={"original_filename": "a.csv"})
    assert names == ["config.yaml", "photos/photo_1.jpg", "point_cloud.csv"]
    with zipfile.ZipFile(buffer) as archive:
        assert archive.namelist()[0] == "project.yaml"
        assert archive.getinfo("photos/photo_1.jpg").compress_type == zipfile.ZIP_STORED and archive.getinfo("config.yaml").compress_type == zipfile.ZIP_DEFLATED
        assert archive.read("point_cloud.csv") == point_cloud.read_bytes()
        manifest = read_project_manifest(archive)
        assert manifest["format_version"] == PROJECT_FORMAT_VERSION and manifest["kind"] == "project" and manifest["contents"] == names and manifest["weather"] == {"original_filename": "a.csv"}
        assert find_project_file(archive, "point_cloud") == "point_cloud.csv" and find_project_file(archive, "weather") is None and find_project_file(archive, "photo_1") is None


def test_project_file_name():
    """Files kept in their own format get the original suffix in lower case, or the default."""
    assert project_file_name("weather", "Freiburg-pvgis-tmy.CSV") == "weather.csv"
    assert project_file_name("weather", "open_meteo.json") == "weather.json"
    assert project_file_name("point_cloud", "scan") == "point_cloud.csv"


def test_project_manifest_checks(caplog):
    """No project.yaml, a newer format version or another kind are errors; listed but missing files only a warning."""
    def archive_with(files):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, text in files.items():
                archive.writestr(name, text)
        return zipfile.ZipFile(buffer)
    with pytest.raises(ValueError, match="is not a project: it has no project.yaml"):
        read_project_manifest(archive_with({"results.json": "{}"}), "results.zip")
    with pytest.raises(ValueError, match="format version 99"):
        read_project_manifest(archive_with({"project.yaml": yaml.safe_dump({"format_version": 99, "kind": "project"})}))
    with pytest.raises(ValueError, match="expected 'project'"):
        read_project_manifest(archive_with({"project.yaml": yaml.safe_dump({"format_version": 1, "kind": "photo_set"})}))
    with caplog.at_level(logging.WARNING):
        read_project_manifest(archive_with({"project.yaml": yaml.safe_dump({"format_version": 1, "kind": "project", "contents": ["config.yaml"]})}), "p.zip")
    assert "missing: config.yaml" in caplog.text
