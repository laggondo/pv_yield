"""Tests of the browser front end: the Python session natively, and a headless Chromium smoke test of the whole page.

The smoke test downloads Pyodide and packages from the CDNs, so it only runs with PV_YIELD_BROWSER_TEST=1 and the
Playwright Python bindings installed (CI job `browser`). The site files are served to the browser by Playwright
itself (`route`), so no web server is needed. Optional environment variables: PV_YIELD_CHROMIUM (path of a Chromium
executable instead of Playwright's own), PV_YIELD_CHROMIUM_ARGS (extra Chromium arguments, space-separated, e.g. to
trust a proxy's CA with --ignore-certificate-errors-spki-list=<hash>) and HTTPS_PROXY (passed to Chromium).
"""

import importlib.util
import io
import json
import mimetypes
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pv_yield_estimator.browser import BrowserSession, config_json_from_yaml, config_yaml_from_json, site_name, site_name_result, site_search, site_search_results, versions, weather_downloads
from pv_yield_estimator.core.photo import camera_orientation_from_device
from test_weather import open_meteo_response

REPOSITORY = Path(__file__).resolve().parents[1]
WEATHER_PATH = REPOSITORY / "data" / "Freiburg-pvgis-tmy.csv"
SKY_PATH = REPOSITORY / "examples" / "sample_obstructed_sky.json"
POINT_CLOUD_PATH = REPOSITORY / "data" / "2026-06-01_22-25-29_red_red.csv"
SAMPLE_CONFIG_PATH = REPOSITORY / "examples" / "sample_config.yaml"
NOMINATIM_ANSWER = '[{"display_name": "Freiburg im Breisgau, Deutschland", "lat": "47.996", "lon": "7.849"}]'
NOMINATIM_REVERSE_ANSWER = '{"display_name": "Rathausplatz, Freiburg im Breisgau, Deutschland", "lat": "47.996", "lon": "7.849"}'
SITE_ORIGIN = "http://localhost:8765"
PANEL = {"tilt_deg": 15.0, "azimuth_deg": 180.0, "area_m2": 1.0, "efficiency": 0.2, "performance_ratio": 0.8}


def load_build_site_module():
    """Import web/build_site.py, which is a script, not part of the package."""
    spec = importlib.util.spec_from_file_location("build_site", REPOSITORY / "web" / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def session():
    """A browser session with the sample data loaded."""
    session = BrowserSession()
    session.load_weather(WEATHER_PATH.read_text(encoding="utf-8"), WEATHER_PATH.name, "pvgis_tmy")
    session.load_obstructed_sky(SKY_PATH.read_text(encoding="utf-8"), SKY_PATH.name)
    return session


def test_session_matches_cli_sample_config(session):
    """Key figures of the browser session equal those of the CLI on the sample data, and changing the panel reuses the irradiation."""
    result = json.loads(session.compute(json.dumps({"weather": {"source": "pvgis_tmy"}, "panel": PANEL})))
    assert result["key_figures"]["sky_view_factor_horizontal"] == pytest.approx(0.5500, abs=1e-4)
    assert "irradiation per sky patch" in result["timings"]
    assert [row["month"] for row in result["monthly_daily_average"]] == list(range(1, 13))
    assert [row["label"] for row in result["orientation_comparison"]][:3] == ["panel", "best obstructed", "best unobstructed"]
    assert result["weather_distance_km"] == pytest.approx(0.0)
    assert set(result["plots"]) == {"sky_hemisphere", "monthly_profiles", "daily_bars", "carpet", "orientation"}
    tilted = json.loads(session.compute(json.dumps({"panel": PANEL | {"tilt_deg": 35.0}})))
    assert "irradiation per sky patch" not in tilted["timings"]
    assert tilted["key_figures"]["annual_total_unobstructed_kwh_m2"] > result["key_figures"]["annual_total_unobstructed_kwh_m2"]


def test_session_summaries_and_errors():
    """Loading returns summaries; computing without weather raises an error; without an obstructed sky description, the sky is free."""
    session = BrowserSession()
    with pytest.raises(ValueError, match="weather data first"):
        session.compute("{}")
    weather = json.loads(session.load_weather(WEATHER_PATH.read_text(encoding="utf-8"), WEATHER_PATH.name, "pvgis_tmy"))
    assert weather["latitude"] == pytest.approx(48.0) and weather["n_hours"] == 8760
    free = json.loads(session.compute(json.dumps({"panel": PANEL, "simulation": {"n_sky_nodes": 100}})))
    assert free["key_figures"]["shading_loss"] == pytest.approx(0.0, abs=1e-12) and free["key_figures"]["sky_view_factor"] == pytest.approx(1.0)
    sky = json.loads(session.load_obstructed_sky(SKY_PATH.read_text(encoding="utf-8"), SKY_PATH.name))
    assert 0 < sky["n_obstructed"] < sky["n_patches"] and sky["method"] == "lidar"
    assert json.loads(session.compute(json.dumps({"panel": PANEL})))["key_figures"]["shading_loss"] > 0.3
    session.clear_obstructed_sky()
    assert json.loads(session.compute(json.dumps({"panel": PANEL})))["key_figures"]["shading_loss"] == pytest.approx(0.0, abs=1e-12)
    assert json.loads(versions())["bokeh"].startswith("3.")


def test_session_exports_and_optimization(session):
    """After computing, the zip holds the export files and the obstructed sky; the PDF report is a PDF; an empty tilt is optimized."""
    result = json.loads(session.compute(json.dumps({"panel": PANEL | {"tilt_deg": None}, "orientation": {"tilt_step_deg": 5, "azimuth_step_deg": 10}})))
    assert result["optimized_angles"] == ["tilt_deg"] and result["key_figures"]["tilt_deg"] in range(0, 91, 5)
    with zipfile.ZipFile(io.BytesIO(session.export_zip())) as archive:
        assert {"results.json", "config.yaml", "hourly.csv", "orientation_grid.csv", "obstructed_sky.json"} <= set(archive.namelist())
        assert json.loads(archive.read("results.json"))["key_figures"] == pytest.approx(result["key_figures"])
    assert session.pdf_report().startswith(b"%PDF")
    assert json.loads(session.obstructed_sky_text())["obstructed"] == json.loads(SKY_PATH.read_text())["obstructed"]


def test_session_lidar_equals_cli_sample(tmp_path):
    """The obstruction computed in the session from the point cloud equals the sample obstructed sky description (made by the CLI with the sample config)."""
    config = json.loads(config_json_from_yaml(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"), SAMPLE_CONFIG_PATH.name))
    summary = json.loads(BrowserSession().compute_obstruction(str(POINT_CLOUD_PATH), POINT_CLOUD_PATH.name, json.dumps(config)))
    expected = json.loads(SKY_PATH.read_text())
    assert summary["n_obstructed"] == sum(expected["obstructed"]) and summary["input_file"] == POINT_CLOUD_PATH.name


def test_session_marking_obstructions():
    """Marking by hand starts from the loaded obstructed sky description, projects its nodes into photos (also from device angles) and replaces it with the edited flags."""
    session = BrowserSession()
    free = json.loads(session.start_editing(100))
    assert free["method"] == "none" and sum(free["obstructed"]) == 0 and len(free["obstructed"]) == len(free["triangles"]) and free["sun_paths"] == []   ### no site, no weather
    ### With a site, the sun paths of the solstices and equinoxes: in June higher in the south than in December.
    paths = {path["label"]: np.array(path["directions"]) for path in json.loads(session.start_editing(100, 48.0, 7.85))["sun_paths"]}
    assert list(paths) == ["21 June", "20 March / 23 September", "21 December"]
    assert paths["21 June"][:, 2].max() == pytest.approx(np.sin(np.radians(90 - 48.0 + 23.44)), abs=0.01) and paths["21 December"][:, 2].max() == pytest.approx(np.sin(np.radians(90 - 48.0 - 23.44)), abs=0.01)
    session.load_obstructed_sky(SKY_PATH.read_text(encoding="utf-8"), SKY_PATH.name)
    start = json.loads(session.start_editing(100))
    assert start["obstructed"] == json.loads(SKY_PATH.read_text())["obstructed"] and start["method"] == "lidar"
    projection = json.loads(session.project_sky(json.dumps({"azimuth_deg": 180, "elevation_deg": 20, "fov_deg": 65, "width": 800, "height": 600})))
    assert len(projection["pixels"]) == len(start["nodes"]) and {marker["label"] for marker in projection["markers"]} == {"SE", "S", "SW", "zenith"}
    assert projection["camera"]["forward"] == pytest.approx([0, -np.cos(np.radians(20)), np.sin(np.radians(20))]) and projection["camera"]["focal_length_px"] == pytest.approx(400 / np.tan(np.radians(32.5)))
    assert projection["sun_paths"] == []                                     ### start_editing without a site, and no weather loaded
    assert len(projection["horizon"]) == 1 and all(y == pytest.approx(300 + 400 / np.tan(np.radians(32.5)) * np.tan(np.radians(20)), abs=0.1) for _, y in projection["horizon"][0][80:100])
    assert next(marker for marker in projection["markers"] if marker["label"] == "S")["x"] == pytest.approx(400)
    from_device = json.loads(session.project_sky(json.dumps({"alpha_deg": 180, "beta_deg": 110, "gamma_deg": 0, "screen_angle_deg": 0, "fov_deg": 65, "width": 600, "height": 800})))
    assert from_device["view"]["azimuth_deg"] == pytest.approx(180) and from_device["view"]["elevation_deg"] == pytest.approx(20)
    edited = start["obstructed"].copy()
    edited[0] = 1 - edited[0]
    summary = json.loads(session.apply_edits(json.dumps(edited), json.dumps(["photo"]), json.dumps({"photos": [projection["view"]]})))
    assert summary["method"] == "lidar+photo" and summary["n_obstructed"] == sum(edited)
    saved = json.loads(session.obstructed_sky_text())
    assert saved["obstructed"] == edited and saved["metadata"]["edits"][0]["n_changed"] == 1 and saved["metadata"]["input_file"] == json.loads(SKY_PATH.read_text())["metadata"]["input_file"]


def test_session_project_round_trip(tmp_path, session):
    """A project saved from the session holds all inputs given, the results and the report; loading it returns the same files (original names, photos and point cloud as files)."""
    session.compute(json.dumps({"panel": PANEL}))
    photo_path, point_cloud_path = tmp_path / "photo.jpg", tmp_path / "point_cloud"
    photo_path.write_bytes(b"\xff\xd8 not really a JPEG")
    point_cloud_path.write_text("x,y,z\n1,2,3\n")
    view = {"azimuth_deg": 180.0, "elevation_deg": 20.0, "roll_deg": 0.0, "fov_deg": 65.0, "width": 400, "height": 300}
    project = {"config": {"site": {"query": "Freiburg"}, "panel": PANEL}, "weather": {"text": WEATHER_PATH.read_text(encoding="utf-8"), "filename": WEATHER_PATH.name, "source": "pvgis_tmy"},
               "obstructed_sky": {"text": SKY_PATH.read_text(encoding="utf-8"), "filename": SKY_PATH.name}, "photos": [{"name": "Photo 1", "taken": "2026-10-09T10:00:00Z", "view": view, "path": str(photo_path)}],
               "point_cloud": {"path": str(point_cloud_path), "filename": "scan.CSV"}, "irradiation": None, "include_results": True}
    project_path = session.save_project(json.dumps(project), str(tmp_path / "project.zip"))
    with zipfile.ZipFile(project_path) as archive:
        names = set(archive.namelist())
        assert archive.read("report.pdf").startswith(b"%PDF")
    assert {"project.yaml", "config.yaml", "weather.csv", "obstructed_sky.json", "photos.json", "photos/photo_1.jpg", "point_cloud.csv", "report.pdf", "results/results.json", "results/hourly.csv"} <= names
    assert "irradiation.json" not in names
    loaded = json.loads(BrowserSession().load_project(project_path, str(tmp_path / "files"), "project.zip"))
    assert loaded["weather"] == project["weather"] and loaded["obstructed_sky"] == project["obstructed_sky"] and loaded["has_results"] and loaded["irradiation"] is None
    assert loaded["config"]["site"]["query"] == "Freiburg" and loaded["config"]["panel"] == PANEL
    assert [(photo["name"], photo["view"]) for photo in loaded["photos"]] == [("Photo 1", view)] and Path(loaded["photos"][0]["path"]).read_bytes() == photo_path.read_bytes()
    assert loaded["point_cloud"]["filename"] == "scan.CSV" and Path(loaded["point_cloud"]["path"]).read_bytes() == point_cloud_path.read_bytes()


def test_session_project_minimal_and_errors(tmp_path):
    """A project with the config only loads with everything else missing; a file that is no zip gives a helpful error."""
    session = BrowserSession()
    project_path = session.save_project(json.dumps({"config": {"panel": {"tilt_deg": 30}}}), str(tmp_path / "project.zip"))
    with zipfile.ZipFile(project_path) as archive:
        assert set(archive.namelist()) == {"project.yaml", "config.yaml"}
    loaded = json.loads(session.load_project(project_path, str(tmp_path / "files")))
    assert loaded["config"]["panel"] == {"tilt_deg": 30} and loaded["weather"] is None and loaded["obstructed_sky"] is None and loaded["photos"] == [] and loaded["point_cloud"] is None and not loaded["has_results"]
    (tmp_path / "no.zip").write_text("not a zip")
    with pytest.raises(ValueError, match="no.zip is not a zip file"):
        session.load_project(str(tmp_path / "no.zip"), str(tmp_path / "files"), "no.zip")


def test_browser_helpers():
    """Site search and weather download URLs, parsing of the search results, config YAML round trip."""
    assert json.loads(site_search("Freiburg")).startswith("https://nominatim.openstreetmap.org/search?q=Freiburg")
    assert json.loads(site_search_results(NOMINATIM_ANSWER))[0]["latitude"] == 47.996
    assert "/reverse?lat=48.000000&lon=7.850000" in json.loads(site_name(48.0, 7.85)) and json.loads(site_name_result(NOMINATIM_REVERSE_ANSWER)).startswith("Rathausplatz")
    downloads = json.loads(weather_downloads(48.0, 7.85, json.dumps({"n_years": 3, "source": "auto"})))
    assert [candidate["service"] for candidate in downloads["candidates"]] == ["open_meteo"] and "re.jrc.ec.europa.eu" in downloads["pvgis_url"] and downloads["pvgis_url"].endswith("&browser=1")
    config = {"panel": {"tilt_deg": None, "azimuth_deg": 180}, "orientation": {"compare": [[30, 90]]}}
    assert json.loads(config_json_from_yaml(config_yaml_from_json(json.dumps(config))))["panel"] == config["panel"]


def test_build_site(tmp_path):
    """The site holds the page, the worker and the package zip."""
    site = load_build_site_module().build_site(tmp_path / "site")
    for path in ("index.html", "app.js", "sky_editor.js", "heading.js", "worker.js", "style.css", "proof/index.html"):
        assert (site / path).is_file(), path
    with zipfile.ZipFile(site / "pv_yield_estimator.zip") as archive:
        names = archive.namelist()
    assert "pv_yield_estimator/browser.py" in names and "pv_yield_estimator/core/sky.py" in names
    assert not (site / "build_site.py").exists()


def swipe_up_over(page, selector):
    """Swipe upwards with one finger over the middle of an element (Chrome DevTools touch events); returns the page's scrollY before and after."""
    page.locator(selector).scroll_into_view_if_needed()
    box = page.locator(selector).bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + min(box["height"], 400) / 2 + 150
    scroll_before = page.evaluate("window.scrollY")
    session = page.context.new_cdp_session(page)
    session.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    for step in range(1, 11):
        session.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y - 30 * step}]})
    session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_timeout(1000)
    return scroll_before, page.evaluate("window.scrollY")


@pytest.mark.skipif(os.environ.get("PV_YIELD_BROWSER_TEST") != "1", reason="headless browser test; set PV_YIELD_BROWSER_TEST=1 (needs Playwright and network access to the Pyodide and Bokeh CDNs)")
def test_browser_page_computes_sample(tmp_path, session):
    """Headless Chromium as a phone: the page loads Pyodide, searches the site and downloads weather (canned answers), computes without obstruction and with the sample files (same key
    figures as the native session), exports zip and PDF, marks obstructions on the sky map, a loaded photo and a camera photo (Chromium's fake camera), computes the obstruction from the
    LiDAR sample, saves all inputs as a project and loads it again, and restores the inputs after a reload; a swipe over a plot scrolls the page; the steps are in tabs."""
    sync_api = pytest.importorskip("playwright.sync_api")
    site = load_build_site_module().build_site(tmp_path / "site")

    def serve_site_file(route):
        """Answer requests to the fake origin from the built site."""
        path = site / (route.request.url.removeprefix(SITE_ORIGIN + "/").split("?")[0] or "index.html")
        if path.is_file():
            route.fulfill(path=path, content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        else:
            route.fulfill(status=404, body=f"not found: {path}")

    ### Canned answers of the site search and the weather service, with the CORS header the real services send.
    cors = {"Access-Control-Allow-Origin": "*"}
    launch_options = {}
    if os.environ.get("PV_YIELD_CHROMIUM"):
        launch_options["executable_path"] = os.environ["PV_YIELD_CHROMIUM"]
    ### Chromium's fake camera (a test pattern) without the permission prompt.
    launch_options["args"] = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] + os.environ.get("PV_YIELD_CHROMIUM_ARGS", "").split()
    if os.environ.get("HTTPS_PROXY"):
        launch_options["proxy"] = {"server": os.environ["HTTPS_PROXY"]}
    expected = json.loads(session.compute(json.dumps({"panel": PANEL})))["key_figures"]
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(**launch_options)
        context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, accept_downloads=True, geolocation={"latitude": 47.99609, "longitude": 7.8494}, permissions=["geolocation"])
        context.route(f"{SITE_ORIGIN}/**", serve_site_file)
        context.route("https://nominatim.openstreetmap.org/**", lambda route: route.fulfill(body=NOMINATIM_REVERSE_ANSWER if "/reverse" in route.request.url else NOMINATIM_ANSWER, content_type="application/json", headers=cors))
        context.route("https://archive-api.open-meteo.com/**", lambda route: route.fulfill(body=open_meteo_response(years=(2024,)), content_type="application/json", headers=cors))
        page = context.new_page()
        messages = []
        page.on("console", lambda message: messages.append(message.text))

        def wait_idle(timeout=300_000):
            """Wait until the page is no longer busy and assert that no error occurred."""
            page.wait_for_function("['ready', 'computed', 'error'].includes(document.body.dataset.state)", timeout=timeout)
            assert page.evaluate("document.body.dataset.state") != "error", page.text_content("#log")

        def download(button, timeout=300_000):
            """Click a button that saves a file; returns the saved file's bytes."""
            with page.expect_download(timeout=timeout) as download_info:
                page.click(button)
            path = tmp_path / download_info.value.suggested_filename
            download_info.value.save_as(path)
            wait_idle()
            return path.read_bytes()

        def tap_canvas(canvas_id, fraction_x, fraction_y):
            """Click a marking canvas (sky map or photo) at a fraction of its size; the locator waits until the canvas stops moving (it scrolls into view smoothly)."""
            canvas = page.locator(f"#{canvas_id}")
            canvas.scroll_into_view_if_needed()
            page.wait_for_timeout(1000)
            box = canvas.bounding_box()
            canvas.click(position={"x": fraction_x * box["width"], "y": fraction_y * box["height"]})

        try:
            page.goto(f"{SITE_ORIGIN}/")
            page.wait_for_function("['ready', 'error'].includes(document.body.dataset.state)", timeout=300_000)
            assert page.evaluate("document.body.dataset.state") == "ready", page.text_content("#log")
            ### Tabs: the first one (site and weather) is shown at the start.
            assert page.locator("#tab-site").is_visible() and page.locator("#tab-panel").is_hidden() and page.get_attribute("#tabs [data-tab=site]", "aria-selected") == "true"
            ### The results tab without weather data says what is missing.
            page.click("#tabs [data-tab=results]")
            assert "need weather data" in page.text_content("#results-empty") and page.locator("#results").is_hidden()
            page.click("#tabs [data-tab=site]")
            ### Site search and weather download.
            page.fill("#site-query", "Freiburg")
            page.click("#site-search")
            wait_idle()
            assert page.input_value("#site-latitude") == "47.99600" and "lat=47.9960" in page.get_attribute("#pvgis-link", "href") and page.get_attribute("#pvgis-link", "href").endswith("browser=1")
            page.click("#weather-download")
            wait_idle()
            assert "Open-Meteo" in page.text_content("#weather-summary")
            ### GPS: the coordinates and the place field (from the reverse search) are updated, the list of found places is hidden.
            page.click("#site-gps")
            wait_idle()
            assert page.input_value("#site-latitude") == "47.99609" and page.input_value("#site-query").startswith("Rathausplatz") and page.locator("#site-results-label").is_hidden()
            ### The sample weather file, without the searched site, so the result matches the native session; first without obstruction (free sky).
            page.fill("#site-latitude", "")
            page.fill("#site-longitude", "")
            page.set_input_files("#weather-file", WEATHER_PATH)
            wait_idle()
            assert "no obstruction" in page.text_content("#sky-summary")
            ### Opening the results tab computes; a new input while it is shown computes again.
            page.click("#tabs [data-tab=results]")
            wait_idle()
            assert page.evaluate("window.pvYieldApp.lastResult.key_figures.shading_loss") == pytest.approx(0.0, abs=1e-12)
            page.set_input_files("#sky-file", SKY_PATH)
            wait_idle()
            assert page.evaluate("document.body.dataset.state") == "computed", page.text_content("#log")
            assert page.locator("#tab-results").is_visible() and page.locator("#tab-site").is_hidden() and page.url.endswith("#/results")
            key_figures = page.evaluate("window.pvYieldApp.lastResult.key_figures")
            for plot in ("sky_hemisphere", "monthly_profiles", "daily_bars", "carpet", "orientation"):
                assert page.locator(f"#plot-{plot} > *").count() > 0, f"plot {plot} not rendered"
            assert page.locator("#orientations tr").count() >= 4
            scroll_before, scroll_after = swipe_up_over(page, "#plot-sky_hemisphere")
            assert scroll_after > scroll_before + 100, f"a swipe over the sky plot did not scroll the page (scrollY {scroll_before} -> {scroll_after})"
            page.screenshot(path=tmp_path / "page.png", full_page=True)
            ### Exports.
            with zipfile.ZipFile(io.BytesIO(download("#export-zip"))) as archive:
                assert {"results.json", "config.yaml", "hourly.csv", "obstructed_sky.json"} <= set(archive.namelist())
            assert download("#export-pdf").startswith(b"%PDF")
            page.click("#menu-button")
            assert b"tilt_deg: 15" in download("#config-save")
            assert page.locator("#menu").is_hidden()
            ### The log below the tab, collapsed at first.
            assert page.locator("#log").is_hidden()
            page.click("#log-details summary")
            assert page.locator("#log").is_visible() and "Computed" in page.text_content("#log")
            ### Marking obstructions by hand, starting from the sample: on the sky map (always shown at the top of the tab), on a loaded photo and on camera photos.
            n_sample_obstructed = sum(json.loads(SKY_PATH.read_text())["obstructed"])
            page.click("#tabs [data-tab=obstruction]")
            assert page.locator("#sky-map-canvas").is_visible() and page.input_value("#n_sky_nodes") == "500"   ### the setting that gives the sample's grid
            tap_canvas("sky-map-canvas", 0.52, 0.48)     ### near the zenith, free in the sample; "Mark" is the default mode
            page.wait_for_function("document.getElementById('sky-summary').textContent.includes('lidar+sky_map')", timeout=60_000)
            assert f": {n_sample_obstructed + 1} of" in page.text_content("#sky-summary")
            ### Only one source section is open at a time (#40).
            page.click("#lidar summary")
            page.click("#photo summary")
            assert page.evaluate("document.getElementById('photo').open && !document.getElementById('lidar').open")
            photo_path = tmp_path / "sky.png"
            pytest.importorskip("matplotlib.pyplot").imsave(photo_path, np.full((300, 400, 3), 0.7))
            page.set_input_files("#photo-file", photo_path)
            wait_idle()
            assert page.input_value("#photo-azimuth_deg") == "180" and page.input_value("#photo-elevation_deg") == "15" and page.locator("#photo-list .photo-item").count() == 1
            assert page.locator("#photo-canvas").is_visible()
            ### The patch in the photo's centre may be obstructed or free in the sample: marking and then freeing it changes it at least once; the switch above the photo is the one above the sky map.
            tap_canvas("photo-canvas", 0.5, 0.5)
            page.click("#mode-free-photo")
            assert "selected" in page.get_attribute("#mode-free", "class")
            tap_canvas("photo-canvas", 0.5, 0.5)
            page.wait_for_function("document.getElementById('sky-summary').textContent.includes('lidar+sky_map+photo')", timeout=60_000)
            page.click("#mode-mark")
            marked_before = page.text_content("#sky-summary")
            ### Aligning the photo: dragging the sky grid to the right with the right mouse button turns the camera to the left (east of south).
            page.locator("#photo-canvas").scroll_into_view_if_needed()
            page.wait_for_timeout(1000)
            box = page.locator("#photo-canvas").bounding_box()
            page.mouse.move(box["x"] + 0.5 * box["width"], box["y"] + 0.5 * box["height"])
            page.mouse.down(button="right")
            page.mouse.move(box["x"] + 0.7 * box["width"], box["y"] + 0.5 * box["height"], steps=5)
            page.mouse.up(button="right")
            page.wait_for_timeout(1000)
            assert float(page.input_value("#photo-azimuth_deg")) < 175 and page.text_content("#sky-summary") == marked_before
            ### The sky map shows the photo merged onto the hemisphere: the grey photo's colour south of the zenith, the background in the north.
            mean_colour = "(x, y) => { const data = document.getElementById('sky-map-canvas').getContext('2d').getImageData(x, y, 10, 10).data; return [0, 1, 2].map(channel => data.filter((_, index) => index % 4 === channel).reduce((sum, value) => sum + value, 0) / 100); }"
            south, north = page.evaluate(f"({mean_colour})(395, 695)"), page.evaluate(f"({mean_colour})(395, 95)")
            assert np.abs(np.subtract(south, north)).max() > 20
            ### Tapping the selected photo's button again hides it.
            page.click("#photo-list .photo-item >> nth=0 >> button >> nth=0")
            assert page.locator("#photo-view").is_hidden()
            page.click("#camera-start")
            wait_idle()
            ### The camera stays open for a series of photos, until "Done".
            for _ in range(2):
                page.click("#camera-shoot")
                wait_idle()
            assert page.locator("#photo-list .photo-item").count() == 3 and page.locator("#camera").is_visible()
            page.click("#camera-stop")
            assert page.locator("#camera").is_hidden() and page.locator("#photo-view").is_hidden()
            photo_set_path = tmp_path / "photo_set.json"
            photo_set_path.write_bytes(download("#photos-save"))
            assert len(json.loads(photo_set_path.read_text())["photos"]) == 3
            ### The trash buttons remove the photos.
            for remaining in (2, 1, 0):
                page.click("#photo-list .photo-item >> nth=0 >> button.remove")
                assert page.locator("#photo-list .photo-item").count() == remaining
            ### "No obstruction" frees the sky.
            page.click("#sky-clear")
            wait_idle()
            assert "no obstruction" in page.text_content("#sky-summary")
            ### Obstruction from the LiDAR sample with the sample config's settings (resolution 500 sky nodes, as set from the loaded sample).
            page.click("#lidar summary")
            page.fill("#scanner_heading_deg", "188.1")
            page.set_input_files("#lidar-file", POINT_CLOUD_PATH)
            page.click("#lidar-compute")
            wait_idle()
            assert f"{n_sample_obstructed} of" in page.text_content("#sky-summary")
            ### Project (#37): with the photo set loaded again, saved with all inputs and the results (computed again, as the inputs changed), then loaded and computed again.
            page.click("#photo summary")
            page.set_input_files("#photo-file", photo_set_path)
            wait_idle()
            assert page.locator("#photo-list .photo-item").count() == 3
            page.click("#menu-button")
            project_path = tmp_path / "project.zip"
            project_path.write_bytes(download("#project-save"))
            with zipfile.ZipFile(project_path) as archive:
                assert {"project.yaml", "config.yaml", "weather.csv", "obstructed_sky.json", "photos.json", "photos/photo_1.jpg", "photos/photo_2.jpg", "point_cloud.csv", "report.pdf", "results/results.json"} <= set(archive.namelist())
            page.set_input_files("#project-file", project_path)
            wait_idle()
            assert page.evaluate("document.body.dataset.state") == "computed" and page.locator("#tab-results").is_visible(), page.text_content("#log")
            assert page.evaluate("window.pvYieldApp.lastResult.key_figures") == pytest.approx(key_figures, rel=1e-6)
            assert page.evaluate("document.getElementById('lidar-file').files[0].name") == POINT_CLOUD_PATH.name and "obstructed_sky_2026" in page.text_content("#sky-summary")
            page.click("#tabs [data-tab=obstruction]")
            assert page.locator("#photo-list .photo-item").count() == 3
            ### The inputs are restored after a reload, also the photos (stored a second after the last change).
            page.wait_for_timeout(2000)
            page.reload()
            page.wait_for_function("['ready', 'computed', 'error'].includes(document.body.dataset.state) && !document.getElementById('project-save').disabled", timeout=300_000)
            assert page.locator("#tab-obstruction").is_visible()                ### the tab is kept in the URL
            assert "Freiburg-pvgis-tmy.csv" in page.text_content("#weather-summary") and "obstructed_sky_2026" in page.text_content("#sky-summary")
            assert page.input_value("#scanner_heading_deg") == "188.1"
            assert page.locator("#photo-list .photo-item").count() == 3
            ### "No obstruction" removes the obstructed sky description again.
            page.click("#tabs [data-tab=obstruction]")
            page.click("#sky-clear")
            wait_idle()
            assert "no obstruction" in page.text_content("#sky-summary") and page.is_disabled("#sky-save")
            ### New project: inputs emptied, the form at its defaults, back on the first tab.
            page.click("#menu-button")
            page.click("#project-new")
            wait_idle()
            assert page.text_content("#weather-summary") == "not loaded" and page.input_value("#scanner_heading_deg") == "0" and page.locator("#tab-site").is_visible()
            assert page.evaluate("window.pvYieldApp.lastResult") is None
            ### Marking on the sky map works after a new project (the session was reset), starting from a free sky.
            page.click("#tabs [data-tab=obstruction]")
            tap_canvas("sky-map-canvas", 0.52, 0.48)
            page.wait_for_function("document.getElementById('sky-summary').textContent.includes(': 1 of')", timeout=60_000)
            assert page.evaluate("document.body.dataset.state") != "error", page.text_content("#log")
            ### Camera with both orientation readings, as on Chrome for Android (synthetic events): the gyroscope's (heading from an arbitrary start) and the compass's, 30° apart.
            ### The shutter waits for a steady compass; the photo gets the compass's heading via the gyroscope; a compass that then disagrees asks for the figure-8 movement and moves the view only slowly.
            page.evaluate("document.getElementById('photo').open = true")
            page.click("#camera-start")
            wait_idle()
            page.evaluate("""() => {
                window.compassAlpha = 130;
                window.sensorTimer = setInterval(() => {
                    window.dispatchEvent(new DeviceOrientationEvent("deviceorientation", { alpha: 100, beta: 80, gamma: 0, absolute: false }));
                    window.dispatchEvent(new DeviceOrientationEvent("deviceorientationabsolute", { alpha: window.compassAlpha, beta: 80, gamma: 0, absolute: true }));
                }, 50);
            }""")
            page.wait_for_timeout(500)
            assert page.is_disabled("#camera-shoot") and "Hold the phone still" in page.text_content("#camera-note")
            page.wait_for_function("!document.getElementById('camera-shoot').disabled", timeout=10_000)
            assert page.locator("#camera-note").is_hidden()
            page.click("#camera-shoot")
            wait_idle()
            page.evaluate("window.compassAlpha = 220")
            page.wait_for_function("document.getElementById('camera-note').textContent.includes('figure 8')", timeout=10_000)
            assert page.is_enabled("#camera-shoot")
            page.click("#camera-shoot")
            wait_idle()
            page.evaluate("clearInterval(window.sensorTimer)")
            page.click("#camera-stop")
            expected_azimuth = camera_orientation_from_device(130, 80, 0)["azimuth_deg"]
            page.click("#photo-list .photo-item >> nth=0 >> button >> nth=0")
            assert float(page.input_value("#photo-azimuth_deg")) == pytest.approx(expected_azimuth, abs=0.1)
            page.click("#photo-list .photo-item >> nth=1 >> button >> nth=0")
            assert abs(float(page.input_value("#photo-azimuth_deg")) - expected_azimuth) < 10       ### the wild compass moved the view only a little
            ### An error shows one line at the top (no traceback); the log has the traceback.
            bad_sky_path = tmp_path / "not_a_sky.json"
            bad_sky_path.write_text('{"kind": "something else"}')
            page.evaluate("document.getElementById('sky-load').open = true")
            page.set_input_files("#sky-file", bad_sky_path)
            page.wait_for_function("document.body.dataset.state === 'error'", timeout=60_000)
            error_text = page.text_content("#error")
            assert "\n" not in error_text and "Traceback" not in error_text and "details in the log" in error_text and "Traceback" in page.text_content("#log")
        finally:
            print("\n".join(messages))
            browser.close()
    assert key_figures == pytest.approx(expected, rel=1e-6)
