// Web Worker running Python via Pyodide, so the page stays responsive during long steps.
// Protocol: the page posts {id, action, args}; the worker answers {id, result} or {id, error}, and posts
// {type: "progress", message} while loading and {type: "log", line} for Python's output (logging, print).
// Results are parsed JSON, or a Uint8Array for file downloads (zip, PDF).

// Pinned versions: Pyodide 314.0.7 bundles numpy, pandas, scipy, pyyaml, pygments (for rich) and bokeh 3.9.0 (Python 3.14); pvlib comes from PyPI.
const PYODIDE_VERSION = "314.0.7";
const PVLIB_VERSION = "0.16.1";
const BUNDLED_PACKAGES = ["micropip", "numpy", "pandas", "scipy", "pyyaml", "bokeh", "pygments", "contourpy"];
const PYPI_PACKAGES = [`pvlib==${PVLIB_VERSION}`, "rich"];

// ?pyodide=<base URL> (passed on from the page URL) loads Pyodide from elsewhere, e.g. a self-hosted copy.
const pyodideBaseUrl = new URL(self.location.href).searchParams.get("pyodide") ?? `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
// Loaded on first use only (PDF report), to keep the first visit smaller.
const REPORT_PACKAGES = ["matplotlib"];
let pyodide = null;
let session = null;
let browser = null;

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
import pv_yield_estimator.browser as browser
from pv_yield_estimator.logging_setup import setup_logging
setup_logging("info")
session = browser.BrowserSession()
`);
  session = pyodide.globals.get("session");
  browser = pyodide.globals.get("browser");
  return JSON.parse(browser.versions());
}

// Copy Python bytes into a Uint8Array for the page.
function bytesResult(pythonBytes) {
  const array = pythonBytes.toJs();
  pythonBytes.destroy();
  return array;
}

// Write a File (from the page) into Pyodide's file system, so Python reads it without a copy as a JavaScript string.
async function writeFileToPython(file, path) {
  pyodide.FS.writeFile(path, new Uint8Array(await file.arrayBuffer()));
  return path;
}

// Remove a directory of temporary files in Pyodide's file system.
function removeDirectory(directory) {
  pyodide.globals.set("directory_to_remove", directory);
  pyodide.runPython("import shutil; shutil.rmtree(directory_to_remove, ignore_errors=True)");
}

// Project zip (#37) from the page's inputs: photos (JPEG bytes) and the point cloud (File) go into Pyodide's file
// system, Python writes the zip there (with the latest results and the PDF report if `project.include_results`);
// returns the zip's bytes.
async function saveProject(project, photos, pointCloud) {
  const directory = "/tmp/project_save";
  pyodide.FS.mkdirTree(directory);
  try {
    project.photos = photos.map(({ bytes, ...photo }, index) => {
      const path = `${directory}/photo_${index + 1}.jpg`;
      pyodide.FS.writeFile(path, bytes);
      return { ...photo, path };
    });
    if (pointCloud) project.point_cloud = { path: await writeFileToPython(pointCloud, `${directory}/point_cloud`), filename: pointCloud.name };
    if (project.include_results) {
      progress(`Loading ${REPORT_PACKAGES.join(", ")} for the PDF report (first time only)`);
      await pyodide.loadPackage(REPORT_PACKAGES);
    }
    return pyodide.FS.readFile(session.save_project(JSON.stringify(project), `${directory}/project.zip`));
  } finally {
    removeDirectory(directory);
  }
}

// Read a project zip (File): the files found, with the photos and the point cloud as bytes.
async function loadProject(file) {
  const directory = "/tmp/project_load";
  pyodide.FS.mkdirTree(directory);
  try {
    const project = JSON.parse(session.load_project(await writeFileToPython(file, `${directory}/project.zip`), `${directory}/files`, file.name));
    for (const photo of project.photos) photo.bytes = pyodide.FS.readFile(photo.path);
    if (project.point_cloud) project.point_cloud.bytes = pyodide.FS.readFile(project.point_cloud.path);
    return project;
  } finally {
    removeDirectory(directory);
  }
}

async function pdfReport() {
  progress(`Loading ${REPORT_PACKAGES.join(", ")} for the PDF report (first time only)`);
  await pyodide.loadPackage(REPORT_PACKAGES);
  return bytesResult(session.pdf_report());
}

const actions = {
  init,
  siteSearchUrl: query => JSON.parse(browser.site_search(query)),
  siteSearchResults: text => JSON.parse(browser.site_search_results(text)),
  siteNameUrl: (latitude, longitude) => JSON.parse(browser.site_name(latitude, longitude)),
  siteNameResult: text => JSON.parse(browser.site_name_result(text)),
  weatherDownloads: (latitude, longitude, weather) => JSON.parse(browser.weather_downloads(latitude, longitude, JSON.stringify(weather))),
  loadWeather: (content, filename, source) => JSON.parse(session.load_weather(content, filename, source)),
  loadObstructedSky: (text, filename) => JSON.parse(session.load_obstructed_sky(text, filename)),
  computeObstruction: async (file, config) => JSON.parse(session.compute_obstruction(await writeFileToPython(file, "/tmp/point_cloud"), file.name, JSON.stringify(config))),
  obstructedSkyText: () => session.obstructed_sky_text(),
  clearObstructedSky: () => session.clear_obstructed_sky(),
  // Pyodide passes null as JsNull, but undefined as None: an unknown site is undefined.
  startEditing: (nSkyNodes, latitude, longitude) => JSON.parse(session.start_editing(nSkyNodes, latitude ?? undefined, longitude ?? undefined)),
  projectSky: view => JSON.parse(session.project_sky(JSON.stringify(view))),
  applyEdits: (obstructed, methods, details) => JSON.parse(session.apply_edits(JSON.stringify(obstructed), JSON.stringify(methods), JSON.stringify(details))),
  compute: config => JSON.parse(session.compute(JSON.stringify(config))),
  configYaml: config => browser.config_yaml_from_json(JSON.stringify(config)),
  configFromYaml: (text, filename) => JSON.parse(browser.config_json_from_yaml(text, filename)),
  exportZip: () => bytesResult(session.export_zip()),
  pdfReport,
  reset: () => session.reset(),
  saveProject,
  loadProject,
};

self.onmessage = async ({ data: { id, action, args } }) => {
  try {
    self.postMessage({ id, result: await actions[action](...args) });
  } catch (error) {
    self.postMessage({ id, error: String(error.message ?? error) });
  }
};
