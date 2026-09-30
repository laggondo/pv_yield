"""Tests of the browser front end: the Python session natively, and a headless Chromium smoke test of the whole page.

The smoke test downloads Pyodide and packages from the CDNs, so it only runs with PV_YIELD_BROWSER_TEST=1 and the
Playwright Python bindings installed (CI job `browser`). The site files are served to the browser by Playwright
itself (`route`), so no web server is needed. Optional environment variables: PV_YIELD_CHROMIUM (path of a Chromium
executable instead of Playwright's own), PV_YIELD_CHROMIUM_ARGS (extra Chromium arguments, space-separated, e.g. to
trust a proxy's CA with --ignore-certificate-errors-spki-list=<hash>) and HTTPS_PROXY (passed to Chromium).
"""

import importlib.util
import json
import mimetypes
import os
import zipfile
from pathlib import Path

import pytest

from pv_yield_estimator.browser import BrowserSession, versions

REPOSITORY = Path(__file__).resolve().parents[1]
WEATHER_PATH = REPOSITORY / "data" / "Freiburg-pvgis-tmy.csv"
SKY_PATH = REPOSITORY / "examples" / "sample_obstructed_sky.json"
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
    """Headless Chromium as a phone: the page loads Pyodide, computes the sample and shows the same key figures as the native session; a swipe over a plot scrolls the page."""
    sync_api = pytest.importorskip("playwright.sync_api")
    site = load_build_site_module().build_site(tmp_path / "site")

    def serve_site_file(route):
        """Answer requests to the fake origin from the built site."""
        path = site / (route.request.url.removeprefix(SITE_ORIGIN + "/").split("?")[0] or "index.html")
        if path.is_file():
            route.fulfill(path=path, content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        else:
            route.fulfill(status=404, body=f"not found: {path}")

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
        context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        context.route(f"{SITE_ORIGIN}/**", serve_site_file)
        page = context.new_page()
        messages = []
        page.on("console", lambda message: messages.append(message.text))
        try:
            page.goto(f"{SITE_ORIGIN}/")
            page.wait_for_function("['ready', 'error'].includes(document.body.dataset.state)", timeout=300_000)
            assert page.evaluate("document.body.dataset.state") == "ready", page.text_content("#log")
            page.click("#load-samples")
            page.wait_for_function("!document.getElementById('compute').disabled || document.body.dataset.state === 'error'", timeout=120_000)
            page.click("#compute")
            page.wait_for_function("['computed', 'error'].includes(document.body.dataset.state)", timeout=300_000)
            assert page.evaluate("document.body.dataset.state") == "computed", page.text_content("#log")
            key_figures = page.evaluate("window.pvYieldApp.lastResult.key_figures")
            for plot in ("sky_hemisphere", "monthly_profiles", "carpet"):
                assert page.locator(f"#plot-{plot} > *").count() > 0, f"plot {plot} not rendered"
            scroll_before, scroll_after = swipe_up_over(page, "#plot-sky_hemisphere")
            assert scroll_after > scroll_before + 100, f"a swipe over the sky plot did not scroll the page (scrollY {scroll_before} -> {scroll_after})"
            page.screenshot(path=tmp_path / "page.png", full_page=True)
        finally:
            print("\n".join(messages))
            browser.close()
    assert key_figures == pytest.approx(expected, rel=1e-6)
