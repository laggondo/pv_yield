// Web Worker running Python via Pyodide, so the page stays responsive during long steps.
// Protocol: the page posts {id, action, args}; the worker answers {id, result} or {id, error}, and posts
// {type: "progress", message} while loading and {type: "log", line} for Python's output (logging, print).

// Pinned versions: Pyodide 314.0.7 bundles numpy, pandas, scipy, pyyaml, pygments (for rich) and bokeh 3.9.0 (Python 3.14); pvlib comes from PyPI.
const PYODIDE_VERSION = "314.0.7";
const PVLIB_VERSION = "0.16.1";
const BUNDLED_PACKAGES = ["micropip", "numpy", "pandas", "scipy", "pyyaml", "bokeh", "pygments"];
const PYPI_PACKAGES = [`pvlib==${PVLIB_VERSION}`, "rich"];

// ?pyodide=<base URL> (passed on from the page URL) loads Pyodide from elsewhere, e.g. a self-hosted copy.
const pyodideBaseUrl = new URL(self.location.href).searchParams.get("pyodide") ?? `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
let pyodide = null;
let session = null;

// Report a loading step to the page.
function progress(message) {
  self.postMessage({ type: "progress", message });
}

// Load Pyodide, the packages and the pv_yield_estimator package (zip next to this file), and create the Python session.
async function init() {
  progress(`Loading Pyodide ${PYODIDE_VERSION} from ${pyodideBaseUrl}`);
  const { loadPyodide } = await import(`${pyodideBaseUrl}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL: pyodideBaseUrl });
  pyodide.setStdout({ batched: line => self.postMessage({ type: "log", line }) });
  pyodide.setStderr({ batched: line => self.postMessage({ type: "log", line }) });
  progress(`Loading bundled packages: ${BUNDLED_PACKAGES.join(", ")}`);
  await pyodide.loadPackage(BUNDLED_PACKAGES);
  // loadPackage only logs failures, so check explicitly (package names in any case, e.g. "Pygments").
  const loadedPackages = Object.keys(pyodide.loadedPackages).map(name => name.toLowerCase());
  const missingPackages = BUNDLED_PACKAGES.filter(name => !loadedPackages.includes(name));
  if (missingPackages.length) throw new Error(`Could not load ${missingPackages.join(", ")} from ${pyodideBaseUrl} (see the browser console)`);
  progress(`Installing from PyPI: ${PYPI_PACKAGES.join(", ")} (first visit: about 30 MB)`);
  pyodide.globals.set("pypi_packages", pyodide.toPy(PYPI_PACKAGES));
  await pyodide.runPythonAsync("import micropip\nawait micropip.install(pypi_packages)");
  progress("Loading the pv_yield_estimator package");
  const response = await fetch("pv_yield_estimator.zip", { cache: "no-cache" });
  if (!response.ok) throw new Error(`Could not fetch pv_yield_estimator.zip: HTTP ${response.status} (build the site with web/build_site.py)`);
  const sitePackages = pyodide.runPython("import site; site.getsitepackages()[0]");
  pyodide.unpackArchive(await response.arrayBuffer(), "zip", { extractDir: sitePackages });
  progress("Importing pvlib, pandas and bokeh");
  pyodide.runPython(`
from pv_yield_estimator.browser import BrowserSession, versions
from pv_yield_estimator.logging_setup import setup_logging
setup_logging("info")
session = BrowserSession()
`);
  session = pyodide.globals.get("session");
  return JSON.parse(pyodide.globals.get("versions")());
}

const actions = {
  init,
  loadWeather: (content, filename, source) => JSON.parse(session.load_weather(content, filename, source)),
  loadObstructedSky: (text, filename) => JSON.parse(session.load_obstructed_sky(text, filename)),
  compute: config => JSON.parse(session.compute(JSON.stringify(config))),
};

self.onmessage = async ({ data: { id, action, args } }) => {
  try {
    self.postMessage({ id, result: await actions[action](...args) });
  } catch (error) {
    self.postMessage({ id, error: String(error.message ?? error) });
  }
};
