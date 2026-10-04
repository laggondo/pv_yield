# Browser app

Static site: plain HTML/JavaScript user interface (`index.html`, `app.js`, `style.css`) with the Python package running in a Web Worker via Pyodide (`worker.js`). The worker loads Pyodide 314.0.7 from the jsDelivr CDN, its bundled packages (numpy, pandas, scipy, pyyaml, bokeh, …), pvlib and rich from PyPI via micropip, and the package itself as `pv_yield_estimator.zip`. Plots come from `pv_yield_estimator.plotting` as Bokeh JSON and are rendered with BokehJS from cdn.bokeh.org (same version as the Python bokeh). The Python entry points are in `src/pv_yield_estimator/browser.py`.

The first visit downloads about 60 MB; later visits use the browser cache.

## Build and run locally

`build_site.py` (standard library only) assembles the site: the files in `web/` (including the `proof/` page) and the package zip. From the repository root:

```
python web/build_site.py                  # writes _site/
python -m http.server 8000 -d _site
```

Then open <http://localhost:8000/>. Rebuild after changing the Python package or files in `web/`.

- `?pyodide=<base URL>` loads Pyodide from elsewhere, e.g. a self-hosted copy of the full distribution.
- Site: place name or address (search with Nominatim, OpenStreetMap), GPS, or coordinates; empty means the weather data's coordinates.
- Weather: downloaded for the site from Open-Meteo (mean of several years; the only free source found that allows requests from a web page, see #8), or a file: PVGIS TMY (CSV/JSON; the page links to the PVGIS file for the site, which PVGIS sends as a download with `browser=1`, so it takes one tap plus picking the file), Open-Meteo JSON or TMY3. The format is detected.
- Obstructed sky description (optional; without it, the sky is free): a JSON file (from the CLI or saved from the page), or computed from a LiDAR point cloud in the page (meant for computers; the file is passed to Python through Pyodide's file system).
- Marking obstructions by hand (photo method, `sky_editor.js`), in a full-screen layer over the page: photos taken with the phone's camera (full screen; camera orientation from the device orientation sensors; camera access needs https or localhost, iOS asks for the motion sensors; hardware keys such as volume up take the photo where the browser passes them to the page), loaded photos, or a map of the sky hemisphere. Tapping or dragging over sky patches marks them; on photos, the sky grid is aligned with the right mouse button (Shift: rotate) and the mouse wheel (field of view), or with two fingers (drag, twist, pinch), helped by the drawn horizon and compass letters. The loaded obstructed sky description is the start state, and every change replaces it right away. Photos can be saved and loaded as a photo set (JSON).
- The irradiation per sky patch is computed on the sky discretization of the obstructed sky description and cached, so changing only the panel entries recomputes just the yield. An empty tilt or azimuth is optimized.
- Files: the weather data, the obstructed sky description and the config (YAML) can be saved and loaded, to move them between devices; results download as a zip (results.json, config.yaml, CSV tables) and as a PDF report (matplotlib is loaded on first use). The loaded inputs and the form are kept in the browser (IndexedDB) for the next visit.

## Hosting

`.github/workflows/pages.yml` builds the site and deploys it to GitHub Pages on every push to `main` (and on manual start). It needs a one-time setting in the repository: Settings → Pages → Build and deployment → Source: **GitHub Actions**. The site is then at <https://laggondo.github.io/pv_yield/>.

## Tests

`tests/test_browser_app.py` tests the Python side natively and, with `PV_YIELD_BROWSER_TEST=1` and the Playwright Python bindings (`playwright-python` from conda-forge, plus `python -m playwright install chromium`), runs a headless Chromium smoke test of the whole page (CI job `browser`): site search, GPS and weather download (answered with canned responses), a free sky, the sample files loaded through the file fields, exports, the PDF report, the LiDAR sample and restoring the inputs after a reload. In the Claude Code container, Chromium needs the proxy's CA; pass it as `PV_YIELD_CHROMIUM_ARGS=--ignore-certificate-errors-spki-list=<SHA-256 of the CA's public key, base64>` together with `PV_YIELD_CHROMIUM=/opt/pw-browsers/chromium`.
