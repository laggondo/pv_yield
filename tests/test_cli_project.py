"""Tests of the CLI with projects (-p): project zips as saved by the browser app, or directories with the same layout; AR recordings to point clouds (#48)."""

import json
import zipfile

import pytest
import yaml

from pv_yield_estimator.cli.main import main
from pv_yield_estimator.core.reconstruction import ReconstructedPoints
from pv_yield_estimator.file_format import to_json_text
from pv_yield_estimator.project import ProjectFiles, write_project_zip
from synthetic_recording import room_recording
from test_cli_pipeline import FAST, SAMPLE_SKY, WEATHER, key_figures


@pytest.fixture(scope="module")
def recording():
    """A short synthetic room recording and its frames."""
    recording, images, scene = room_recording(n_frames=16)
    return recording, images, scene


def recording_project(path, recording, images, config=None):
    """Write a project zip as the browser app saves it after an AR recording: recording.json, the frames and the config."""
    contents = {"recording.json": to_json_text(recording), **{frame["file"]: image for frame, image in zip(recording["frames"], images)}}
    if config is not None:
        contents["config.yaml"] = yaml.safe_dump(config)
    write_project_zip(path, contents)
    return path


def test_obstruction_from_recording_directory(tmp_path, recording):
    """`obstruction` with a directory holding recording.json and its frames writes the point cloud (CSV) and the check plot; settings come from sky_obstruction.ar_video."""
    recording, images, scene = recording
    directory = tmp_path / "recording"
    (directory / "recording").mkdir(parents=True)
    (directory / "recording.json").write_text(to_json_text(recording))
    for frame, image in zip(recording["frames"], images):
        (directory / frame["file"]).write_bytes(image)
    main(["obstruction", str(directory), "-o", str(tmp_path / "points.csv"), "--plot", str(tmp_path / "points.png"), "-m", "sky_obstruction.ar_video.max_features=1500"])
    points = ReconstructedPoints.from_csv_text((tmp_path / "points.csv").read_text())
    assert len(points.table) > 500 and points.metadata["settings"]["max_features"] == 1500 and (tmp_path / "points.png").stat().st_size > 10_000
    main(["obstruction", str(directory / "recording.json"), "-o", str(tmp_path / "points2.csv")])
    assert (tmp_path / "points2.csv").is_file()


def test_obstruction_from_project_zip(tmp_path, recording):
    """`obstruction -p` takes the recording from a project zip and its config, writes the point cloud and the config into the zip, and keeps the other files."""
    recording, images, scene = recording
    project = recording_project(tmp_path / "project.zip", recording, images, {"format_version": 1, "panel": {"tilt_deg": 25}, "sky_obstruction": {"ar_video": {"min_frames": 2}}})
    main(["obstruction", "-p", str(project), "-m", "sky_obstruction.ar_video.min_triangulation_angle_deg=3"])
    with zipfile.ZipFile(project) as archive:
        names = set(archive.namelist())
        assert {"project.yaml", "recording.json", "recording/frame_1.jpg", "recording/frame_16.jpg", "reconstructed_points.csv", "config.yaml"} <= names
        manifest = yaml.safe_load(archive.read("project.yaml"))
        assert "reconstructed_points.csv" in manifest["contents"] and archive.read("recording/frame_3.jpg") == images[2]
        config = yaml.safe_load(archive.read("config.yaml"))
        assert config["panel"]["tilt_deg"] == 25 and config["sky_obstruction"]["ar_video"] == {"min_frames": 2, "min_triangulation_angle_deg": 3}
        points = ReconstructedPoints.from_csv_text(archive.read("reconstructed_points.csv").decode())
    assert points.table["triangulation_angle_deg"].min() >= 3 and len(points.table) > 300
    with pytest.raises(ValueError, match="Give -o/--output or a project"):
        main(["obstruction", str(tmp_path)])


def test_run_into_new_project_then_yield(tmp_path):
    """`run -p` with a new zip stores the inputs, the intermediate files, the results and the report; `yield -p` recomputes from the project alone, as `yield -d` does."""
    project = tmp_path / "site.zip"
    main(["run", "-p", str(project), "-s", str(SAMPLE_SKY), "-w", str(WEATHER), *FAST])
    with zipfile.ZipFile(project) as archive:
        names = set(archive.namelist())
        manifest = yaml.safe_load(archive.read("project.yaml"))
        results = json.loads(archive.read("results/results.json"))
    assert {"config.yaml", "weather.csv", "obstructed_sky.json", "irradiation.json", "report.pdf", "results/results.json", "results/hourly.csv"} <= names
    assert manifest["weather"] == {"file": "weather.csv", "original_filename": WEATHER.name, "source": "auto"}
    main(["yield", "-p", str(project), "-d", str(tmp_path / "tilt30"), "-m", "panel.tilt_deg=30"])
    with zipfile.ZipFile(project) as archive:
        tilt30 = json.loads(archive.read("results/results.json"))
        assert yaml.safe_load(archive.read("config.yaml"))["panel"]["tilt_deg"] == 30
    assert tilt30["config"]["orientation"]["tilt_step_deg"] == 10                        ### the project's config came first
    assert tilt30["key_figures"] == pytest.approx(key_figures(tmp_path / "tilt30"))
    assert tilt30["key_figures"]["annual_total_obstructed_kwh_m2"] != pytest.approx(results["key_figures"]["annual_total_obstructed_kwh_m2"])
    main(["run", "-p", str(project), "-d", str(tmp_path / "again")])                       ### all inputs from the project
    assert key_figures(tmp_path / "again") == pytest.approx(tilt30["key_figures"])


def test_project_directory(tmp_path):
    """A directory with the project layout works like a zip: `irradiation -p` stores the weather file and writes irradiation.json and project.yaml."""
    directory = tmp_path / "project"
    main(["irradiation", str(WEATHER), "-p", str(directory), "-m", "simulation.n_sky_nodes=100"])
    assert {path.name for path in directory.iterdir()} == {"project.yaml", "config.yaml", "weather.csv", "irradiation.json"}
    files = ProjectFiles(directory)
    assert files.details["weather"]["original_filename"] == WEATHER.name and files.find("weather") == "weather.csv"
    main(["yield", "-p", str(directory), *FAST])
    assert (directory / "results" / "results.json").is_file() and (directory / "report.pdf").is_file()


def test_project_files_replace_and_remove(tmp_path):
    """A file kept in its own format replaces one of another suffix; changes are only written by save; a non-zip file is refused."""
    path = tmp_path / "p.zip"
    files = ProjectFiles(path)
    files.write_stem_file("weather", "a.csv", "csv text")
    files.write("config.yaml", "panel: {}\n")
    assert not path.exists()
    files.save()
    files = ProjectFiles(path)
    files.write_stem_file("weather", "b.json", "{}")
    assert files.read_text("config.yaml") == "panel: {}\n" and files.find("weather") == "weather.json"
    files.save()
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == ["config.yaml", "project.yaml", "weather.json"]
    (tmp_path / "x.zip").write_text("not a zip")
    with pytest.raises(ValueError, match="is not a zip file"):
        ProjectFiles(tmp_path / "x.zip")


def test_browser_keeps_points_from_cli(tmp_path, recording):
    """A project with the point cloud from the CLI goes through the browser app's load and save unchanged."""
    from pv_yield_estimator.browser import BrowserSession
    recording, images, _ = recording
    project = recording_project(tmp_path / "project.zip", recording, images)
    main(["obstruction", "-p", str(project)])
    session = BrowserSession()
    loaded = json.loads(session.load_project(str(project), str(tmp_path / "files"), "project.zip"))
    assert loaded["reconstructed_points"]["text"].startswith("# format_version: 1")
    saved = session.save_project(json.dumps({"config": loaded["config"], "recording": loaded["recording"], "reconstructed_points": loaded["reconstructed_points"]}), str(tmp_path / "saved.zip"))
    with zipfile.ZipFile(project) as original, zipfile.ZipFile(saved) as archive:
        assert archive.read("reconstructed_points.csv") == original.read("reconstructed_points.csv")
