# Browser app

Static site: plain HTML/JavaScript user interface (`index.html`, `app.js`, `style.css`) with the Python package running in a Web Worker via Pyodide (`worker.js`). The worker loads Pyodide 314.0.7 from the jsDelivr CDN, its bundled packages (numpy, pandas, scipy, pyyaml, bokeh, …), pvlib and rich from PyPI via micropip, and the package itself as `pv_yield_estimator.zip`. Plots come from `pv_yield_estimator.plotting` as Bokeh JSON and are rendered with BokehJS from cdn.bokeh.org (same version as the Python bokeh). The Python entry points are in `src/pv_yield_estimator/browser.py`.

The first visit downloads about 60 MB; later visits use the browser cache.

## Build and run locally

`build_site.py` (standard library only) assembles the site: the files in `web/` (including the `proof/` page), the package zip and the sample files for the "Load sample data" button (`data/Freiburg-pvgis-tmy.csv`, `examples/sample_obstructed_sky.json`). From the repository root:

```
python web/build_site.py                  # writes _site/
python -m http.server 8000 -d _site
```

Then open <http://localhost:8000/>. Rebuild after changing the Python package or files in `web/`.

- `?pyodide=<base URL>` loads Pyodide from elsewhere, e.g. a self-hosted copy of the full distribution.
- Inputs: a weather file (PVGIS TMY CSV or TMY3) and an obstructed sky description as written by the CLI (`pv-yield-estimator obstruction ...`). The irradiation per sky patch is computed on the sky discretization of that file and cached, so changing only the panel entries recomputes just the yield.

## Hosting

`.github/workflows/pages.yml` builds the site and deploys it to GitHub Pages on every push to `main` (and on manual start). It needs a one-time setting in the repository: Settings → Pages → Build and deployment → Source: **GitHub Actions**. The site is then at <https://laggondo.github.io/pv_yield/>.

## Tests

`tests/test_browser_app.py` tests the Python side natively and, with `PV_YIELD_BROWSER_TEST=1` and the Playwright Python bindings (`playwright-python` from conda-forge, plus `python -m playwright install chromium`), runs a headless Chromium smoke test of the whole page on the sample data (CI job `browser`). In the Claude Code container, Chromium needs the proxy's CA; pass it as `PV_YIELD_CHROMIUM_ARGS=--ignore-certificate-errors-spki-list=<SHA-256 of the CA's public key, base64>` together with `PV_YIELD_CHROMIUM=/opt/pw-browsers/chromium`.
