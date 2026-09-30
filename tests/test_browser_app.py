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

import pytest

from pv_yield_estimator.browser import BrowserSession, config_json_from_yaml, config_yaml_from_json, site_search, site_search_results, versions, weather_downloads
from test_weather import open_meteo_response

REPOSITORY = Path(__file__).resolve().parents[1]
WEATHER_PATH = REPOSITORY / "data" / "Freiburg-pvgis-tmy.csv"
SKY_PATH = REPOSITORY / "examples" / "sample_obstructed_sky.json"
POINT_CLOUD_PATH = REPOSITORY / "data" / "2026-06-01_22-25-29_red_red.csv"
SAMPLE_CONFIG_PATH = REPOSITORY / "examples" / "sample_config.yaml"
NOMINATIM_ANSWER = '[{"display_name": "Freiburg im Breisgau, Deutschland", "lat": "47.996", "lon": "7.849"}]'
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
    """Loading returns summaries; computing without inputs raises an error naming what is missing."""
    session = BrowserSession()
    with pytest.raises(ValueError, match="weather loaded: False"):
        session.compute("{}")
    weather = json.loads(session.load_weather(WEATHER_PATH.read_text(encoding="utf-8"), WEATHER_PATH.name, "pvgis_tmy"))
    assert weather["latitude"] == pytest.approx(48.0) and weather["n_hours"] == 8760
    sky = json.loads(session.load_obstructed_sky(SKY_PATH.read_text(encoding="utf-8"), SKY_PATH.name))
    assert 0 < sky["n_obstructed"] < sky["n_patches"] and sky["method"] == "lidar"
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


def test_browser_helpers():
    """Site search and weather download URLs, parsing of the search results, config YAML round trip."""
    assert json.loads(site_search("Freiburg")).startswith("https://nominatim.openstreetmap.org/search?q=Freiburg")
    assert json.loads(site_search_results(NOMINATIM_ANSWER))[0]["latitude"] == 47.996
    downloads = json.loads(weather_downloads(48.0, 7.85, json.dumps({"n_years": 3, "source": "auto"})))
    assert [candidate["service"] for candidate in downloads["candidates"]] == ["open_meteo"] and "re.jrc.ec.europa.eu" in downloads["pvgis_url"] and downloads["pvgis_url"].endswith("&browser=1")
    config = {"panel": {"tilt_deg": None, "azimuth_deg": 180}, "orientation": {"compare": [[30, 90]]}}
    assert json.loads(config_json_from_yaml(config_yaml_from_json(json.dumps(config))))["panel"] == config["panel"]


def test_build_site(tmp_path):
    """The site holds the page, the worker, the package zip and the sample files."""
    site = load_build_site_module().build_site(tmp_path / "site")
    for path in ("index.html", "app.js", "worker.js", "style.css", "proof/index.html", "samples/Freiburg-pvgis-tmy.csv", "samples/sample_obstructed_sky.json"):
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
    """Headless Chromium as a phone: the page loads Pyodide, searches the site and downloads weather (canned answers), computes the sample with the same key
    figures as the native session, exports zip and PDF, computes the obstruction from the LiDAR sample, and restores the inputs after a reload; a swipe over a plot scrolls the page."""
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
    if os.environ.get("PV_YIELD_CHROMIUM_ARGS"):
        launch_options["args"] = os.environ["PV_YIELD_CHROMIUM_ARGS"].split()
    if os.environ.get("HTTPS_PROXY"):
        launch_options["proxy"] = {"server": os.environ["HTTPS_PROXY"]}
    expected = json.loads(session.compute(json.dumps({"panel": PANEL})))["key_figures"]
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(**launch_options)
        context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, accept_downloads=True)
        context.route(f"{SITE_ORIGIN}/**", serve_site_file)
        context.route("https://nominatim.openstreetmap.org/**", lambda route: route.fulfill(body=NOMINATIM_ANSWER, content_type="application/json", headers=cors))
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

        try:
            page.goto(f"{SITE_ORIGIN}/")
            page.wait_for_function("['ready', 'error'].includes(document.body.dataset.state)", timeout=300_000)
            assert page.evaluate("document.body.dataset.state") == "ready", page.text_content("#log")
            ### Site search and weather download.
            page.fill("#site-query", "Freiburg")
            page.click("#site-search")
            wait_idle()
            assert page.input_value("#site-latitude") == "47.99600" and "lat=47.9960" in page.get_attribute("#pvgis-link", "href") and page.get_attribute("#pvgis-link", "href").endswith("browser=1")
            page.click("#weather-download")
            wait_idle()
            assert "Open-Meteo" in page.text_content("#weather-summary")
            ### Sample data, without the searched site, so the result matches the native session.
            page.fill("#site-latitude", "")
            page.fill("#site-longitude", "")
            page.click("#load-samples")
            wait_idle()
            page.click("#compute")
            wait_idle()
            assert page.evaluate("document.body.dataset.state") == "computed", page.text_content("#log")
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
            assert b"tilt_deg: 15" in download("#config-save")
            ### Obstruction from the LiDAR sample with the sample config's settings.
            page.click("#lidar summary")
            page.fill("#scanner_heading_deg", "188.1")
            page.set_input_files("#lidar-file", POINT_CLOUD_PATH)
            page.click("#lidar-compute")
            wait_idle()
            assert f"{sum(json.loads(SKY_PATH.read_text())['obstructed'])} of" in page.text_content("#sky-summary")
            ### The inputs are restored after a reload.
            page.reload()
            page.wait_for_function("['ready', 'error'].includes(document.body.dataset.state) && !document.getElementById('compute').disabled", timeout=300_000)
            assert "Freiburg-pvgis-tmy.csv" in page.text_content("#weather-summary") and "obstructed_sky_2026" in page.text_content("#sky-summary")
            assert page.input_value("#scanner_heading_deg") == "188.1"
        finally:
            print("\n".join(messages))
            browser.close()
    assert key_figures == pytest.approx(expected, rel=1e-6)
