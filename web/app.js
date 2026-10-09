// User interface of the browser app; the computation runs in Python in a Web Worker (worker.js).
// The page has tabs for the main steps (site and weather, obstruction, panel, results) and a menu (☰) for actions that
// are not steps (#36); the status light stays in the top bar. Opening the results tab computes the results if they are
// missing or out of date (inputs or form changed). The page state is in document.body.dataset.state: loading, ready, busy, computed or error (used by the smoke test and
// the red/green status light); progress messages and timings go to the log at the bottom.
// Loaded inputs, the photos and the form are kept in IndexedDB for the next visit (not the point cloud, which may be large); files can be saved and loaded to move them
// between devices, one by one or all at once as a project zip (#37).

import { createSkyEditor } from "./sky_editor.js";

const NO_SKY_SUMMARY = "none: no obstruction (free sky)";
const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// Form fields stored for the next visit and filled from a loaded config.
const FORM_FIELDS = ["site-query", "site-latitude", "site-longitude", "weather-years", "weather-source", "tilt_deg", "azimuth_deg", "area_m2", "efficiency", "performance_ratio", "compare",
  "scanner_heading_deg", "offset-east", "offset-north", "offset-up", "min_points", "n_sky_nodes", "camera_fov_deg"];
// Enter in these fields opens the results tab (and so computes).
const ENTER_SHOWS_RESULTS = ["tilt_deg", "azimuth_deg", "area_m2", "efficiency", "performance_ratio", "compare"];

const pageStart = performance.now();
const element = id => document.getElementById(id);
const worker = new Worker(`worker.js${location.search}`, { type: "module" });
const pendingCalls = new Map();
let nextCallId = 0;
let pythonReady = false;
// Loaded inputs as file content, kept to re-parse (weather format change), to save as files and to store for the next visit.
// The irradiation per sky patch is only kept to save it again, if a loaded project has it (the page computes its own).
const inputs = { weather: null, sky: null, irradiation: null };        // {text, filename, source?}
// Config entries without a form field, from a loaded config file; the form's entries are merged over them.
let importedConfig = {};
// Latest result, for inspection in the browser console and the smoke test.
window.pvYieldApp = { lastResult: null };
// Counts changes of the loaded inputs (weather, obstructed sky description); with the config, it tells whether the
// latest result is up to date: results.computedKey is the key of the latest result, results.attemptedKey that of the
// latest computation, also a failed one (not repeated until something changes). Plots computed while the results tab
// is hidden are drawn when it is shown (hidden plots get no size).
let inputsVersion = 0;
const results = { computedKey: null, attemptedKey: null, pendingPlots: null };

const STATE_TEXTS = { loading: "Loading Python and packages ...", ready: "Ready", busy: "Working ...", computed: "Ready", error: "Error" };

// Set the page state and the status text next to the red/green light.
function setState(state) {
  document.body.dataset.state = state;
  element("status-text").textContent = STATE_TEXTS[state];
}

// ---- Tabs and menu ----

// Tab panels: the main steps (tab bar) and the about page (menu).
const TABS = ["site", "obstruction", "panel", "results", "about"];

// Show one tab panel (unknown names show the first); scrolls to the top when the tab changes. The results tab computes
// the results if needed.
function showTab(name) {
  if (!TABS.includes(name)) name = TABS[0];
  if (!element(`tab-${name}`).hidden) return;
  for (const tab of TABS) element(`tab-${tab}`).hidden = tab !== name;
  for (const button of document.querySelectorAll("#tabs [data-tab]")) button.setAttribute("aria-selected", String(button.dataset.tab === name));
  window.scrollTo(0, 0);
  // Drawn while hidden, the sky map and photo get line widths for the wrong size.
  if (name === "obstruction") editor.redraw();
  updateResultsIfShown();
}

// Show a tab right away and keep it in the URL (#/name), so a reload stays on it and the back button returns to the
// previous tab. The hash has a slash, so it never scrolls to an element with that id.
function openTab(name) {
  showTab(name);
  if (location.hash !== `#/${name}`) location.hash = `/${name}`;
}

function setMenuOpen(open) {
  element("menu").hidden = !open;
  element("menu-button").setAttribute("aria-expanded", String(open));
}

// Add a line to the log panel.
function appendLog(line) {
  element("log").textContent += line + "\n";
}

// Add a progress message with the time since page load to the log and the console.
function report(message) {
  appendLog(`[${((performance.now() - pageStart) / 1000).toFixed(1)} s] ${message}`);
  console.log(message);
}

worker.onmessage = ({ data }) => {
  if (data.type === "progress") return report(data.message);
  if (data.type === "log") return appendLog(data.line);
  const { resolve, reject } = pendingCalls.get(data.id);
  pendingCalls.delete(data.id);
  if ("error" in data) reject(new Error(data.error));
  else resolve(data.result);
};
worker.onerror = event => fail(new Error(`Worker error: ${event.message ?? event}`));

// Call an action of the worker and return a promise of its result.
function call(action, ...args) {
  return new Promise((resolve, reject) => {
    const id = nextCallId++;
    pendingCalls.set(id, { resolve, reject });
    worker.postMessage({ id, action, args });
  });
}

// Show an error below the status light, log it and mark the page state.
function fail(error) {
  const message = `Error: ${error.message ?? error}`;
  report(message);
  console.error(message);
  element("error").textContent = `${message}\n(Details: log at the bottom of the page.)`;
  element("error").hidden = false;
  setState("error");
  updateButtons();
}

// Enable the buttons that can be used in the current state.
function updateButtons() {
  const idle = pythonReady && document.body.dataset.state !== "busy";
  for (const id of ["site-search", "weather-download", "config-save", "project-new", "project-save"]) element(id).disabled = !idle;
  element("lidar-compute").disabled = !idle || !element("lidar-file").files.length;
  element("camera-start").disabled = !idle || !editor.isStarted;
  element("weather-save").disabled = !inputs.weather;
  element("sky-save").disabled = !inputs.sky;
  element("sky-clear").disabled = !idle || !inputs.sky;
  element("export-zip").disabled = element("export-pdf").disabled = !idle || !window.pvYieldApp.lastResult;
}

// Run an action while the page is marked busy; errors are reported, not rethrown. `finalState` may be a function,
// evaluated after the action.
async function whileBusy(action, finalState = "ready") {
  setState("busy");
  element("error").hidden = true;
  updateButtons();
  try {
    await action();
    setState(typeof finalState === "function" ? finalState() : finalState);
  } catch (error) {
    fail(error);
  }
  updateButtons();
  updateResultsIfShown();
}

// Load a classic script (BokehJS) and resolve when it has run.
function loadScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Could not load script ${url}`));
    document.head.appendChild(script);
  });
}

async function fetchText(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new Error(`Could not fetch ${url}: ${error.message} (offline, or the service does not allow requests from this page)`);
  }
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status} ${(await response.text()).slice(0, 300)}`);
  return response.text();
}

// Offer content (text or bytes) as a file download; on phones it lands in the downloads folder.
function saveFile(filename, content, type = "application/octet-stream") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  report(`Saved ${filename}`);
}

const format = (value, digits) => Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const percent = value => `${format(100 * value, 1)} %`;

// ---- Storage for the next visit (IndexedDB; the page works without it, e.g. in private windows) ----

let databasePromise = null;
function database() {
  databasePromise ??= new Promise((resolve, reject) => {
    const request = indexedDB.open("pv_yield_estimator", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("inputs");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return databasePromise;
}

// Run one request on the store; resolves with its result.
async function storeRequest(mode, makeRequest) {
  const transaction = (await database()).transaction("inputs", mode);
  const request = makeRequest(transaction.objectStore("inputs"));
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// Store a value for the next visit; storage failures only get logged, as the page works without storage.
async function remember(key, value) {
  try {
    await storeRequest("readwrite", store => store.put(value, key));
  } catch (error) {
    report(`Could not store ${key} in the browser: ${error.message ?? error}`);
  }
}

async function recall(key) {
  try {
    return await storeRequest("readonly", store => store.get(key));
  } catch (error) {
    report(`Could not read stored ${key}: ${error.message ?? error}`);
    return undefined;
  }
}

function formValues() {
  return Object.fromEntries(FORM_FIELDS.map(id => [id, element(id).value]));
}

function setFormValues(values) {
  for (const [id, value] of Object.entries(values ?? {})) if (FORM_FIELDS.includes(id) && value !== undefined && value !== null) element(id).value = value;
}

// ---- Config: form <-> config dict ----

// A form field's label and tab, for error messages, e.g. "Area (m²)" (tab Panel).
function fieldName(id) {
  const label = element(id).closest("label")?.firstChild?.textContent.trim() || id;
  const tab = element(id).closest(".tab-panel")?.id.replace("tab-", "");
  const tabTitle = document.querySelector(`#tabs [data-tab="${tab}"]`)?.textContent;
  return `"${label}"${tabTitle ? ` (tab ${tabTitle})` : ""}`;
}

// A number from a form field; empty gives null (e.g. "optimize" or "from the weather data").
function numberOrNull(id) {
  const text = element(id).value.trim();
  if (text === "") return null;
  const value = Number(text);
  if (Number.isNaN(value)) throw new Error(`${fieldName(id)} is not a number: ${JSON.stringify(text)}`);
  return value;
}

function requiredNumber(id) {
  const value = numberOrNull(id);
  if (value === null) throw new Error(`${fieldName(id)} is empty`);
  return value;
}

// "30/180, 90/180" -> [[30, 180], [90, 180]].
function parseCompare(text) {
  return text.split(/[,;]/).map(entry => entry.trim()).filter(Boolean).map(entry => {
    const parts = entry.split("/").map(Number);
    if (parts.length !== 2 || parts.some(Number.isNaN)) throw new Error(`Orientation to compare must be tilt/azimuth, e.g. 30/180; got ${JSON.stringify(entry)}`);
    return parts;
  });
}

function mergeDeep(base, update) {
  const merged = structuredClone(base);
  for (const [key, value] of Object.entries(update)) {
    merged[key] = value && typeof value === "object" && !Array.isArray(value) && typeof merged[key] === "object" && merged[key] !== null ? mergeDeep(merged[key], value) : value;
  }
  return merged;
}

// The config as a (partial) config dict: the loaded config file's entries with the form's entries over them;
// missing entries take the defaults of the Python functions.
function currentConfig() {
  return mergeDeep(importedConfig, {
    site: { query: element("site-query").value.trim() || null, latitude: numberOrNull("site-latitude"), longitude: numberOrNull("site-longitude") },
    weather: { source: element("weather-source").value, n_years: requiredNumber("weather-years") },
    sky_obstruction: { method: "lidar", lidar: { scanner_heading_deg: requiredNumber("scanner_heading_deg"), panel_offset_m: ["offset-east", "offset-north", "offset-up"].map(requiredNumber), min_points: requiredNumber("min_points") },
      photo: { fov_deg: requiredNumber("camera_fov_deg") } },
    panel: { tilt_deg: numberOrNull("tilt_deg"), azimuth_deg: numberOrNull("azimuth_deg"), area_m2: requiredNumber("area_m2"), efficiency: requiredNumber("efficiency"), performance_ratio: requiredNumber("performance_ratio") },
    orientation: { compare: parseCompare(element("compare").value) },
    simulation: { n_sky_nodes: requiredNumber("n_sky_nodes") },
  });
}

// Fill the form from a config dict; entries without a form field are kept for `currentConfig`.
function applyConfig(config) {
  const value = (section, key) => config[section]?.[key];
  const set = (id, entry) => { element(id).value = entry ?? ""; };
  if (value("site", "query") || value("site", "name")) set("site-query", value("site", "query") || value("site", "name"));
  set("site-latitude", value("site", "latitude"));
  set("site-longitude", value("site", "longitude"));
  if (value("weather", "source")) element("weather-source").value = value("weather", "source");
  if (value("weather", "n_years")) set("weather-years", value("weather", "n_years"));
  for (const key of ["tilt_deg", "azimuth_deg"]) set(key, value("panel", key));
  for (const key of ["area_m2", "efficiency", "performance_ratio"]) if (value("panel", key) !== undefined) set(key, value("panel", key));
  if (value("orientation", "compare")) set("compare", value("orientation", "compare").map(pair => pair.join("/")).join(", "));
  const lidar = config.sky_obstruction?.lidar ?? {};
  if (lidar.scanner_heading_deg !== undefined) set("scanner_heading_deg", lidar.scanner_heading_deg);
  if (lidar.panel_offset_m) ["offset-east", "offset-north", "offset-up"].forEach((id, index) => set(id, lidar.panel_offset_m[index]));
  if (lidar.min_points !== undefined) set("min_points", lidar.min_points);
  if (config.sky_obstruction?.photo?.fov_deg !== undefined) set("camera_fov_deg", config.sky_obstruction.photo.fov_deg);
  if (value("simulation", "n_sky_nodes")) set("n_sky_nodes", value("simulation", "n_sky_nodes"));
  importedConfig = config;
  updatePvgisLink();
}

// ---- Inputs ----

async function loadWeather(text, filename, source = "auto") {
  inputs.weather = null;
  element("weather-summary").textContent = "loading ...";
  const summary = await call("loadWeather", text, filename, source);
  inputs.weather = { text, filename, source: summary.source };
  inputsVersion++;
  element("weather-source").value = source;
  element("weather-summary").textContent = `${filename}: ${summary.name}, ${format(summary.latitude, 3)}° N, ${format(summary.longitude, 3)}° E, ${format(summary.altitude, 0)} m; ${summary.n_hours} hours; annual GHI ${format(summary.annual_ghi_kwh_m2, 0)}, DNI ${format(summary.annual_dni_kwh_m2, 0)}, DHI ${format(summary.annual_dhi_kwh_m2, 0)} kWh/m²`;
  report(`Weather loaded: ${filename} (${summary.source})`);
  await remember("weather", inputs.weather);
  if (editor.isStarted) await editor.reload();      // sun paths for the weather data's site, if the form has none
}

function showSkySummary(summary, filename) {
  element("sky-summary").textContent = `${filename}: ${summary.n_obstructed} of ${summary.n_patches} sky patches obstructed (${percent(summary.obstructed_solid_angle_fraction)} of the solid angle), horizontal sky view factor ${format(summary.sky_view_factor_horizontal, 3)}; method ${summary.method || "?"}${summary.input_file ? ` from ${summary.input_file}` : ""}`;
}

async function loadObstructedSky(text, filename) {
  inputs.sky = null;
  element("sky-summary").textContent = "loading ...";
  const summary = await call("loadObstructedSky", text, filename);
  showSkySummary(summary, filename);
  // The resolution field takes the setting that gives the loaded description's grid (none for other grids).
  if (summary.n_sky_nodes !== null) element("n_sky_nodes").value = summary.n_sky_nodes;
  inputs.sky = { text, filename };
  inputsVersion++;
  report(`Obstructed sky description loaded: ${filename}`);
  await remember("sky", inputs.sky);
  if (editor.isStarted) await editor.reload();
}

async function computeObstruction() {
  const file = element("lidar-file").files[0];
  inputs.sky = null;
  element("sky-summary").textContent = `computing from ${file.name} (${format(file.size / 1e6, 1)} MB) ...`;
  report(`Computing the obstructed sky from ${file.name}`);
  const summary = await call("computeObstruction", file, currentConfig());
  const filename = `obstructed_sky_${file.name.replace(/\.[^.]*$/, "")}.json`;
  inputs.sky = { text: await call("obstructedSkyText"), filename };
  inputsVersion++;
  showSkySummary(summary, filename);
  report(`Obstructed sky computed from ${file.name}`);
  await remember("sky", inputs.sky);
  if (editor.isStarted) await editor.reload();
}

// The obstructed sky description marked by hand (photos, sky map) replaces the loaded one; kept like a loaded file.
async function skyMarked(summary) {
  const filename = `obstructed_sky_${(summary.method || "marked").replace(/\+/g, "_")}.json`;
  inputs.sky = { text: await call("obstructedSkyText"), filename };
  inputsVersion++;
  showSkySummary(summary, filename);
  updateButtons();
  await remember("sky", inputs.sky);
}

const editor = createSkyEditor({ element, call, report, fail, defaultFov: () => requiredNumber("camera_fov_deg"), nSkyNodes: () => requiredNumber("n_sky_nodes"),
  site: () => ({ latitude: numberOrNull("site-latitude"), longitude: numberOrNull("site-longitude") }), onApplied: skyMarked,
  onPhotosChanged: async () => remember("photos", await editor.photoFiles()) });

// The site's coordinates from the form, or an error asking for them.
function siteCoordinates() {
  const latitude = numberOrNull("site-latitude"), longitude = numberOrNull("site-longitude");
  if (latitude === null || longitude === null) throw new Error("Set the site first: search for a place, use GPS or enter latitude and longitude");
  return { latitude, longitude };
}

function setSite(latitude, longitude) {
  element("site-latitude").value = Number(latitude).toFixed(5);
  element("site-longitude").value = Number(longitude).toFixed(5);
  updatePvgisLink();
  remember("form", formValues());
  if (editor.isStarted) editor.reload().catch(fail);      // sun paths for the new site
}

// Point the PVGIS link to the site's typical year, if the site is set; PVGIS sends it as a file download, so no new tab
// (without a site, the link opens the PVGIS website in a new tab).
async function updatePvgisLink() {
  const latitude = numberOrNull("site-latitude"), longitude = numberOrNull("site-longitude");
  if (!pythonReady || latitude === null || longitude === null) return;
  element("pvgis-link").href = (await call("weatherDownloads", latitude, longitude, { n_years: numberOrNull("weather-years") ?? 10 })).pvgis_url;
  element("pvgis-link").removeAttribute("target");
}

// Places found by the last site search, offered in the list below the search field.
let foundPlaces = [];

// Fill the place field with the address at the site's coordinates (reverse search), e.g. after GPS or typed coordinates;
// the list of found places no longer applies. A failed lookup only gets logged: the coordinates are what counts.
async function describeSite() {
  const latitude = numberOrNull("site-latitude"), longitude = numberOrNull("site-longitude");
  if (!pythonReady || latitude === null || longitude === null) return;
  foundPlaces = [];
  element("site-results-label").hidden = true;
  element("site-query").value = "";
  try {
    const name = await call("siteNameResult", await fetchText(await call("siteNameUrl", latitude, longitude)));
    element("site-query").value = name;
    report(`Site at ${latitude}, ${longitude}: ${name || "no address found"}`);
  } catch (error) {
    report(`Could not look up the address of the site: ${error.message ?? error}`);
  }
  remember("form", formValues());
}

async function searchSite() {
  const query = element("site-query").value.trim();
  const places = await call("siteSearchResults", await fetchText(await call("siteSearchUrl", query)));
  if (!places.length) throw new Error(`No place found for ${JSON.stringify(query)}; try another spelling or enter the coordinates`);
  foundPlaces = places;
  element("site-results").innerHTML = places.map((place, index) => `<option value="${index}">${place.name.replace(/</g, "&lt;")}</option>`).join("");
  element("site-results-label").hidden = false;
  setSite(places[0].latitude, places[0].longitude);
  report(`Site: ${places[0].name} (${places.length} places found)`);
}

function locateByGps() {
  return new Promise((resolve, reject) => {
    if (!navigator.geolocation) return reject(new Error("This browser has no location service"));
    navigator.geolocation.getCurrentPosition(position => {
      setSite(position.coords.latitude, position.coords.longitude);
      report(`Site from GPS: ${position.coords.latitude.toFixed(5)}, ${position.coords.longitude.toFixed(5)} (±${Math.round(position.coords.accuracy)} m)`);
      describeSite().then(resolve);
    }, error => reject(new Error(`Location not available: ${error.message}`)), { enableHighAccuracy: true, timeout: 30_000 });
  });
}

async function downloadWeather() {
  const { latitude, longitude } = siteCoordinates();
  const { candidates } = await call("weatherDownloads", latitude, longitude, { n_years: requiredNumber("weather-years") });
  const candidate = candidates[0];
  report(`Downloading weather data: ${candidate.description}`);
  element("weather-summary").textContent = `downloading from ${candidate.service} ...`;
  await loadWeather(await fetchText(candidate.url), candidate.filename, candidate.weather_source);
}

// ---- Project: all files in one zip (#37) ----

// File name of a saved project: the place (first part of the address) and the date, e.g. Freiburg_im_Breisgau_2026-10-09.pvproject.zip.
function projectFilename() {
  const place = element("site-query").value.split(",")[0].normalize("NFKD").replace(/[\u0300-\u036f]/g, "").replace(/[^A-Za-z0-9-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40);
  return `${place || "pv_yield"}_${new Date().toISOString().slice(0, 10)}.pvproject.zip`;
}

// Save all inputs that are there (config, weather, obstructed sky description, photos, point cloud) as one zip, with
// the results and the PDF report if they can be computed (computed now if out of date), so they match the inputs.
async function saveProject() {
  let includeResults = false;
  if (inputs.weather) {
    try {
      if (!resultsUpToDate()) await compute();
      includeResults = true;
    } catch (error) {
      report(`Saving the project without results, as computing them failed: ${error.message ?? error}`);
    }
  }
  const project = { config: currentConfig(), weather: inputs.weather, obstructed_sky: inputs.sky, irradiation: inputs.irradiation, include_results: includeResults };
  const pointCloud = element("lidar-file").files[0] ?? null;
  report(`Saving the project${pointCloud ? ` with the point cloud ${pointCloud.name} (${format(pointCloud.size / 1e6, 1)} MB)` : ""} ...`);
  saveFile(projectFilename(), await call("saveProject", project, await editor.photoFiles(), pointCloud), "application/zip");
}

// Empty the inputs and set the form to the page's defaults (also in the browser's storage), e.g. before loading a project.
async function resetInputs() {
  await call("reset");
  Object.assign(inputs, { weather: null, sky: null, irradiation: null });
  inputsVersion++;
  importedConfig = {};
  for (const id of FORM_FIELDS) {
    const field = element(id);
    field.value = field.tagName === "SELECT" ? ([...field.options].find(option => option.defaultSelected) ?? field.options[0]).value : field.defaultValue;
  }
  for (const id of ["weather-file", "sky-file", "lidar-file"]) element(id).value = "";
  foundPlaces = [];
  element("site-results-label").hidden = true;
  element("weather-summary").textContent = "not loaded";
  element("sky-summary").textContent = NO_SKY_SUMMARY;
  await editor.setPhotos([]);
  // The reset session forgot the start state of marking: start again from a free sky.
  if (editor.isStarted) await editor.reload();
  window.pvYieldApp.lastResult = null;
  Object.assign(results, { computedKey: null, attemptedKey: null, pendingPlots: null });
  showResultsMessage(NO_RESULTS_MESSAGE);
  await Promise.all([remember("form", formValues()), remember("config", {}), remember("weather", null), remember("sky", null)]);
}

// Start a new project: empty inputs, the form at its defaults (also for the next visit).
async function newProject() {
  await resetInputs();
  openTab("site");
  report("New project: inputs emptied, form at its defaults");
}

// Load a project: replaces the current inputs; files missing in the project leave their inputs empty or at the
// defaults. If the project holds results, the results tab opens and computes them again from the project's inputs.
async function loadProject(file) {
  if (!pythonReady) throw new Error("Python is still loading; load the project when the page is ready");
  const project = await call("loadProject", file);
  await resetInputs();
  if (project.config) {
    applyConfig(project.config);
    await remember("config", project.config);
  }
  await remember("form", formValues());
  if (project.weather) await loadWeather(project.weather.text, project.weather.filename, project.weather.source);
  if (project.obstructed_sky) await loadObstructedSky(project.obstructed_sky.text, project.obstructed_sky.filename);
  if (project.photos.length) await editor.setPhotos(project.photos);
  if (project.point_cloud) {
    // A file input can only be set through a DataTransfer; the point cloud is then used like a picked file.
    const transfer = new DataTransfer();
    transfer.items.add(new File([project.point_cloud.bytes], project.point_cloud.filename));
    element("lidar-file").files = transfer.files;
    element("lidar").open = true;
  }
  if (project.irradiation) inputs.irradiation = project.irradiation;
  report(`Project loaded: ${file.name} (saved ${project.created} with version ${project.program_version}): ${project.contents.join(", ")}`);
  openTab(project.has_results && inputs.weather ? "results" : "site");
}

// ---- Results ----

function showKeyFigures(result) {
  const figures = result.key_figures;
  const optimized = result.optimized_angles.length ? ` (optimized: ${result.optimized_angles.map(name => name.replace("_deg", "")).join(", ")})` : "";
  const pairs = [
    ["Radiation on the panel, total (kWh/m²)", "annual_total", "_kwh_m2", 1],
    ["Radiation on the panel, direct (kWh/m²)", "annual_direct", "_kwh_m2", 1],
    ["Radiation on the panel, diffuse (kWh/m²)", "annual_diffuse", "_kwh_m2", 1],
    ["PV yield (kWh)", "annual_yield", "_kwh", 1],
    ["Specific yield (kWh/kWp)", "specific_yield", "_kwh_kwp", 0],
  ];
  const singles = [
    ["Shading loss, total (%)", format(100 * figures.shading_loss, 1)],
    ["Shading loss, direct (%)", format(100 * figures.direct_shading_loss, 1)],
    ["Sky view factor of the panel", format(figures.sky_view_factor, 3)],
    ["Sky view factor, horizontal", format(figures.sky_view_factor_horizontal, 3)],
    ["Rated power (kWp)", format(figures.rated_power_kwp, 3)],
  ];
  const distance = result.weather_distance_km === null ? "" : `; weather data ${format(result.weather_distance_km, 1)} km away`;
  element("key-figures").innerHTML = `<caption>Panel: tilt ${format(figures.tilt_deg, 0)}°, azimuth ${format(figures.azimuth_deg, 0)}°${optimized}; site ${format(result.site.latitude, 3)}°, ${format(result.site.longitude, 3)}°${distance}</caption>`
    + `<tr><th>Annual</th><th class="number">unobstructed</th><th class="number">obstructed</th></tr>`
    + pairs.map(([label, prefix, suffix, digits]) => `<tr><td>${label}</td><td class="number">${format(figures[`${prefix}_unobstructed${suffix}`], digits)}</td><td class="number">${format(figures[`${prefix}_obstructed${suffix}`], digits)}</td></tr>`).join("")
    + singles.map(([label, value]) => `<tr><td>${label}</td><td></td><td class="number">${value}</td></tr>`).join("");
}

function showMonthly(rows) {
  element("monthly").innerHTML = `<tr><th>Month</th><th class="number">radiation unobstructed (kWh/m²/d)</th><th class="number">radiation obstructed (kWh/m²/d)</th><th class="number">yield unobstructed (kWh/d)</th><th class="number">yield obstructed (kWh/d)</th></tr>`
    + rows.map(row => `<tr><td>${MONTH_NAMES[row.month - 1]}</td><td class="number">${format(row.total_unobstructed, 2)}</td><td class="number">${format(row.total_obstructed, 2)}</td><td class="number">${format(row.yield_unobstructed, 3)}</td><td class="number">${format(row.yield_obstructed, 3)}</td></tr>`).join("");
}

function showOrientations(rows) {
  element("orientations").innerHTML = `<tr><th></th><th class="number">tilt (°)</th><th class="number">azimuth (°)</th><th class="number">radiation unobstructed (kWh/m²)</th><th class="number">radiation obstructed (kWh/m²)</th><th class="number">yield obstructed (kWh)</th><th class="number">specific yield obstructed (kWh/kWp)</th><th class="number">shading loss (%)</th></tr>`
    + rows.map(row => `<tr><td>${row.label}</td><td class="number">${format(row.tilt_deg, 0)}</td><td class="number">${format(row.azimuth_deg, 0)}</td><td class="number">${format(row.annual_total_unobstructed_kwh_m2, 1)}</td><td class="number">${format(row.annual_total_obstructed_kwh_m2, 1)}</td><td class="number">${format(row.annual_yield_obstructed_kwh, 1)}</td><td class="number">${format(row.specific_yield_obstructed_kwh_kwp, 0)}</td><td class="number">${format(100 * row.shading_loss, 1)}</td></tr>`).join("");
}

async function showPlots(plots) {
  for (const [name, item] of Object.entries(plots)) {
    const container = element(`plot-${name}`);
    container.innerHTML = "";
    await Bokeh.embed.embed_item(item, container.id);
  }
}

const NO_RESULTS_MESSAGE = "No results yet: set the site and the weather (first tab), optionally the obstruction and the panel; the results are computed when you open this tab.";
const MISSING_WEATHER_MESSAGE = "The results need weather data: download it or load a file in the tab Site & weather.";

// Instead of the results, a message (e.g. which input is missing).
function showResultsMessage(message) {
  element("results").hidden = true;
  element("results-empty").textContent = message;
  element("results-empty").hidden = false;
}

// The key of the current inputs and form; the latest result is up to date if it was computed with the same key.
function resultsKey() {
  return JSON.stringify([currentConfig(), inputsVersion]);
}

function resultsUpToDate() {
  return window.pvYieldApp.lastResult !== null && resultsKey() === results.computedKey;
}

// Compute the results for the current inputs and show them (the plots once the results tab is shown).
async function compute() {
  const config = currentConfig();
  const key = resultsKey();
  results.attemptedKey = key;
  report(`Computing: tilt ${config.panel.tilt_deg ?? "optimized"}°, azimuth ${config.panel.azimuth_deg ?? "optimized"}° ...`);
  await remember("form", formValues());
  let result;
  try {
    result = await call("compute", config);
  } catch (error) {
    window.pvYieldApp.lastResult = null;
    showResultsMessage("Computing the results failed: see the error at the top. They are computed again when an input changes.");
    throw error;
  }
  window.pvYieldApp.lastResult = result;
  results.computedKey = key;
  element("results").hidden = false;
  element("results-empty").hidden = true;
  showKeyFigures(result);
  showMonthly(result.monthly_daily_average);
  showOrientations(result.orientation_comparison);
  results.pendingPlots = result.plots;
  await drawPendingPlots();
  report(`Computed (${Object.entries(result.timings).map(([step, seconds]) => `${step} ${seconds.toFixed(1)} s`).join(", ")})`);
}

// Draw the plots of the latest result if the results tab is shown.
async function drawPendingPlots() {
  if (!results.pendingPlots || element("tab-results").hidden) return;
  const plots = results.pendingPlots;
  results.pendingPlots = null;
  await showPlots(plots);
}

// While the results tab is shown and the page is idle: compute if the results are missing or out of date (once per
// change of the inputs), or say which input is missing; draw plots computed while the tab was hidden.
function updateResultsIfShown() {
  if (!pythonReady || element("tab-results").hidden || document.body.dataset.state === "busy") return;
  if (!inputs.weather) return showResultsMessage(MISSING_WEATHER_MESSAGE);
  let key;
  try {
    key = resultsKey();
  } catch (error) {
    showResultsMessage("The results can't be computed: see the error at the top.");
    return fail(error);
  }
  if (window.pvYieldApp.lastResult && key === results.computedKey) return drawPendingPlots().catch(fail);
  if (key !== results.attemptedKey) whileBusy(compute, "computed");
}

// Restore the inputs and form of the last visit, if stored.
async function restoreStoredInputs() {
  const [form, weather, sky, config, photos] = await Promise.all([recall("form"), recall("weather"), recall("sky"), recall("config"), recall("photos")]);
  if (config) importedConfig = config;
  setFormValues(form);
  if (weather) await loadWeather(weather.text, weather.filename, weather.source ?? "auto");
  if (sky) await loadObstructedSky(sky.text, sky.filename);
  if (photos?.length) await editor.setPhotos(photos);
  if (form || weather || sky || photos?.length) report(`Restored the inputs of the last visit${photos?.length ? ` (${photos.length} photos)` : ""}`);
  updatePvgisLink();
  await editor.reload();
}

// ---- Event handlers ----

for (const button of document.querySelectorAll("[data-tab]")) button.addEventListener("click", () => openTab(button.dataset.tab));
window.addEventListener("hashchange", () => showTab(location.hash.replace(/^#\/?/, "")));
element("menu-button").addEventListener("click", () => setMenuOpen(element("menu").hidden));
// A tap anywhere else closes the menu; a tap on a menu entry acts first (a file chooser opens from its label).
document.addEventListener("click", event => {
  if (!element("menu").hidden && !element("menu-button").contains(event.target)) setTimeout(() => setMenuOpen(false));
});
document.addEventListener("keydown", event => { if (event.key === "Escape") setMenuOpen(false); });

for (const id of FORM_FIELDS) element(id).addEventListener("change", () => remember("form", formValues()));
element("site-search").addEventListener("click", () => whileBusy(searchSite));
element("site-query").addEventListener("keydown", event => { if (event.key === "Enter" && !element("site-search").disabled) element("site-search").click(); });
element("site-results").addEventListener("change", event => setSite(foundPlaces[event.target.value].latitude, foundPlaces[event.target.value].longitude));
element("site-gps").addEventListener("click", () => whileBusy(locateByGps));
for (const id of ["site-latitude", "site-longitude", "weather-years"]) element(id).addEventListener("change", () => updatePvgisLink().catch(fail));
for (const id of ["site-latitude", "site-longitude"]) element(id).addEventListener("change", () => {
  describeSite();
  if (editor.isStarted) editor.reload().catch(fail);      // sun paths for the new site
});
element("weather-download").addEventListener("click", () => whileBusy(downloadWeather));
element("weather-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  // A new file's format is detected; the format menu only re-reads the loaded file in another format.
  if (file) await whileBusy(async () => loadWeather(await file.text(), file.name, "auto"));
});
element("weather-source").addEventListener("change", async event => {
  if (inputs.weather) await whileBusy(() => loadWeather(inputs.weather.text, inputs.weather.filename, event.target.value));
});
element("weather-save").addEventListener("click", () => saveFile(inputs.weather.filename, inputs.weather.text, "text/plain"));
element("sky-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (file) await whileBusy(async () => loadObstructedSky(await file.text(), file.name));
});
element("lidar-file").addEventListener("change", updateButtons);
element("lidar-compute").addEventListener("click", () => whileBusy(computeObstruction));
element("sky-save").addEventListener("click", () => saveFile(inputs.sky.filename, inputs.sky.text, "application/json"));
async function clearObstructedSky() {
  await call("clearObstructedSky");
  inputs.sky = null;
  inputsVersion++;
  element("sky-summary").textContent = NO_SKY_SUMMARY;
  element("sky-file").value = "";
  report("Obstructed sky description removed: computing without obstruction");
  await remember("sky", null);
  if (editor.isStarted) await editor.reload();
}

element("sky-clear").addEventListener("click", () => whileBusy(clearObstructedSky));
// A new resolution starts again from a free sky; with an obstructed sky description, only after a confirmation.
element("n_sky_nodes").addEventListener("focus", event => { event.target.dataset.previous = event.target.value; });
element("n_sky_nodes").addEventListener("change", event => whileBusy(async () => {
  if (inputs.sky && !confirm(`A new resolution replaces the obstructed sky description (${inputs.sky.filename}) by a free sky. Continue?`)) {
    event.target.value = event.target.dataset.previous ?? event.target.value;
    await remember("form", formValues());
    return;
  }
  if (inputs.sky) await clearObstructedSky();
  else await editor.reload();
}));
// The camera starts right in the tap's handler: iOS grants the motion sensors only then.
element("camera-start").addEventListener("click", () => whileBusy(editor.startCamera));
element("camera-shoot").addEventListener("click", () => whileBusy(editor.takePhoto));
element("camera-stop").addEventListener("click", () => editor.stopCamera());
element("photo-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (file) await whileBusy(() => editor.loadPhotoFile(file));
  event.target.value = "";
});
element("photos-save").addEventListener("click", () => {
  if (editor.hasPhotos) saveFile("pv_yield_photos.json", editor.photoSetText(), "application/json");
  else report("No photos to save");
});
for (const id of ENTER_SHOWS_RESULTS) {
  element(id).addEventListener("keydown", event => { if (event.key === "Enter") openTab("results"); });
}
element("config-save").addEventListener("click", () => whileBusy(async () => saveFile("pv_yield_config.yaml", await call("configYaml", currentConfig()), "text/yaml")));
element("config-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (file) await whileBusy(async () => {
    const config = await call("configFromYaml", await file.text(), file.name);
    applyConfig(config);
    await remember("config", config);
    await remember("form", formValues());
    report(`Config loaded: ${file.name}`);
  });
  event.target.value = "";
});
const readyOrComputed = () => window.pvYieldApp.lastResult ? "computed" : "ready";
element("project-new").addEventListener("click", () => whileBusy(newProject, readyOrComputed));
element("project-save").addEventListener("click", () => whileBusy(saveProject, readyOrComputed));
element("project-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (file) await whileBusy(() => loadProject(file), readyOrComputed);
  event.target.value = "";
});
element("export-zip").addEventListener("click", () => whileBusy(async () => saveFile("pv_yield_results.zip", await call("exportZip"), "application/zip"), "computed"));
element("export-pdf").addEventListener("click", () => whileBusy(async () => saveFile("pv_yield_report.pdf", await call("pdfReport"), "application/pdf"), "computed"));

showTab(location.hash.replace(/^#\/?/, ""));
try {
  updateButtons();
  const versions = await call("init");
  report(`Python ready: ${Object.entries(versions).map(([name, version]) => `${name} ${version}`).join(", ")}`);
  // BokehJS must match the Python bokeh version exactly.
  report(`Loading BokehJS ${versions.bokeh}`);
  await loadScript(`https://cdn.bokeh.org/bokeh/release/bokeh-${versions.bokeh}.min.js`);
  pythonReady = true;
  await whileBusy(restoreStoredInputs);
  if (document.body.dataset.state === "ready") report("Ready: set the site and load or download weather data, optionally load or compute an obstructed sky description, then open the results");
} catch (error) {
  fail(error);
}
