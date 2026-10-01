// User interface of the browser app; the computation runs in Python in a Web Worker (worker.js).
// The page state is in document.body.dataset.state: loading, ready, busy, computed or error (used by the smoke test and
// the red/green status light); progress messages and timings go to the log at the bottom.
// Loaded inputs and the form are kept in IndexedDB for the next visit; files can be saved and loaded to move them
// between devices.

const SAMPLE_WEATHER = { url: "samples/Freiburg-pvgis-tmy.csv", source: "pvgis_tmy" };
const SAMPLE_SKY = { url: "samples/sample_obstructed_sky.json" };
const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// Form fields stored for the next visit and filled from a loaded config.
const FORM_FIELDS = ["site-query", "site-latitude", "site-longitude", "weather-years", "weather-source", "tilt_deg", "azimuth_deg", "area_m2", "efficiency", "performance_ratio", "compare",
  "scanner_heading_deg", "offset-east", "offset-north", "offset-up", "min_points", "n_sky_nodes"];
const ENTER_COMPUTES = ["tilt_deg", "azimuth_deg", "area_m2", "efficiency", "performance_ratio", "compare"];

const pageStart = performance.now();
const element = id => document.getElementById(id);
const worker = new Worker(`worker.js${location.search}`, { type: "module" });
const pendingCalls = new Map();
let nextCallId = 0;
let pythonReady = false;
// Loaded inputs as file content, kept to re-parse (weather format change), to save as files and to store for the next visit.
const inputs = { weather: null, sky: null };        // {text, filename, source?}
// Config entries without a form field, from a loaded config file; the form's entries are merged over them.
let importedConfig = {};
// Latest result, for inspection in the browser console and the smoke test.
window.pvYieldApp = { lastResult: null };

const STATE_TEXTS = { loading: "Loading Python and packages ...", ready: "Ready", busy: "Working ...", computed: "Ready", error: "Error" };

// Set the page state and the status text next to the red/green light.
function setState(state) {
  document.body.dataset.state = state;
  element("status-text").textContent = STATE_TEXTS[state];
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
  element("error").textContent = message;
  element("error").hidden = false;
  setState("error");
  updateButtons();
}

// Enable the buttons that can be used in the current state.
function updateButtons() {
  const idle = pythonReady && document.body.dataset.state !== "busy";
  for (const id of ["load-samples", "site-search", "weather-download", "config-save"]) element(id).disabled = !idle;
  element("lidar-compute").disabled = !idle || !element("lidar-file").files.length;
  element("compute").disabled = !idle || !inputs.weather || !inputs.sky;
  element("weather-save").disabled = !inputs.weather;
  element("sky-save").disabled = !inputs.sky;
  element("export-zip").disabled = element("export-pdf").disabled = !idle || !window.pvYieldApp.lastResult;
}

// Run an action while the page is marked busy; errors are reported, not rethrown.
async function whileBusy(action, finalState = "ready") {
  setState("busy");
  element("error").hidden = true;
  updateButtons();
  try {
    await action();
    setState(finalState);
  } catch (error) {
    fail(error);
  }
  updateButtons();
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

// A number from a form field; empty gives null (e.g. "optimize" or "from the weather data").
function numberOrNull(id) {
  const text = element(id).value.trim();
  if (text === "") return null;
  const value = Number(text);
  if (Number.isNaN(value)) throw new Error(`Field ${id} is not a number: ${JSON.stringify(text)}`);
  return value;
}

function requiredNumber(id) {
  const value = numberOrNull(id);
  if (value === null) throw new Error(`Field ${id} is empty`);
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
    site: { latitude: numberOrNull("site-latitude"), longitude: numberOrNull("site-longitude") },
    weather: { source: element("weather-source").value, n_years: requiredNumber("weather-years") },
    sky_obstruction: { method: "lidar", lidar: { scanner_heading_deg: requiredNumber("scanner_heading_deg"), panel_offset_m: ["offset-east", "offset-north", "offset-up"].map(requiredNumber), min_points: requiredNumber("min_points") } },
    panel: { tilt_deg: numberOrNull("tilt_deg"), azimuth_deg: numberOrNull("azimuth_deg"), area_m2: requiredNumber("area_m2"), efficiency: requiredNumber("efficiency"), performance_ratio: requiredNumber("performance_ratio") },
    orientation: { compare: parseCompare(element("compare").value) },
    simulation: { n_sky_nodes: requiredNumber("n_sky_nodes") },
  });
}

// Fill the form from a config dict; entries without a form field are kept for `currentConfig`.
function applyConfig(config) {
  const value = (section, key) => config[section]?.[key];
  const set = (id, entry) => { element(id).value = entry ?? ""; };
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
  element("weather-source").value = source;
  element("weather-summary").textContent = `${filename}: ${summary.name}, ${format(summary.latitude, 3)}° N, ${format(summary.longitude, 3)}° E, ${format(summary.altitude, 0)} m; ${summary.n_hours} hours; annual GHI ${format(summary.annual_ghi_kwh_m2, 0)}, DNI ${format(summary.annual_dni_kwh_m2, 0)}, DHI ${format(summary.annual_dhi_kwh_m2, 0)} kWh/m²`;
  report(`Weather loaded: ${filename} (${summary.source})`);
  await remember("weather", inputs.weather);
}

function showSkySummary(summary, filename) {
  element("sky-summary").textContent = `${filename}: ${summary.n_obstructed} of ${summary.n_patches} sky patches obstructed (${percent(summary.obstructed_solid_angle_fraction)} of the solid angle), horizontal sky view factor ${format(summary.sky_view_factor_horizontal, 3)}; method ${summary.method || "?"}${summary.input_file ? ` from ${summary.input_file}` : ""}`;
}

async function loadObstructedSky(text, filename) {
  inputs.sky = null;
  element("sky-summary").textContent = "loading ...";
  showSkySummary(await call("loadObstructedSky", text, filename), filename);
  inputs.sky = { text, filename };
  report(`Obstructed sky description loaded: ${filename}`);
  await remember("sky", inputs.sky);
}

async function computeObstruction() {
  const file = element("lidar-file").files[0];
  inputs.sky = null;
  element("sky-summary").textContent = `computing from ${file.name} (${format(file.size / 1e6, 1)} MB) ...`;
  report(`Computing the obstructed sky from ${file.name}`);
  const summary = await call("computeObstruction", file, currentConfig());
  const filename = `obstructed_sky_${file.name.replace(/\.[^.]*$/, "")}.json`;
  inputs.sky = { text: await call("obstructedSkyText"), filename };
  showSkySummary(summary, filename);
  report(`Obstructed sky computed from ${file.name}`);
  await remember("sky", inputs.sky);
}

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

async function compute() {
  const config = currentConfig();
  report(`Computing: tilt ${config.panel.tilt_deg ?? "optimized"}°, azimuth ${config.panel.azimuth_deg ?? "optimized"}° ...`);
  await remember("form", formValues());
  const result = await call("compute", config);
  window.pvYieldApp.lastResult = result;
  element("results").hidden = false;
  showKeyFigures(result);
  showMonthly(result.monthly_daily_average);
  showOrientations(result.orientation_comparison);
  await showPlots(result.plots);
  report(`Computed (${Object.entries(result.timings).map(([step, seconds]) => `${step} ${seconds.toFixed(1)} s`).join(", ")})`);
}

// Restore the inputs and form of the last visit, if stored.
async function restoreStoredInputs() {
  const [form, weather, sky, config] = await Promise.all([recall("form"), recall("weather"), recall("sky"), recall("config")]);
  if (config) importedConfig = config;
  setFormValues(form);
  if (weather) await loadWeather(weather.text, weather.filename, form?.["weather-source"] ?? "auto");
  if (sky) await loadObstructedSky(sky.text, sky.filename);
  if (form || weather || sky) report("Restored the inputs of the last visit");
  updatePvgisLink();
}

// ---- Event handlers ----

for (const id of FORM_FIELDS) element(id).addEventListener("change", () => remember("form", formValues()));
element("site-search").addEventListener("click", () => whileBusy(searchSite));
element("site-query").addEventListener("keydown", event => { if (event.key === "Enter" && !element("site-search").disabled) element("site-search").click(); });
element("site-results").addEventListener("change", event => setSite(foundPlaces[event.target.value].latitude, foundPlaces[event.target.value].longitude));
element("site-gps").addEventListener("click", () => whileBusy(locateByGps));
for (const id of ["site-latitude", "site-longitude", "weather-years"]) element(id).addEventListener("change", () => updatePvgisLink().catch(fail));
for (const id of ["site-latitude", "site-longitude"]) element(id).addEventListener("change", () => describeSite());
element("weather-download").addEventListener("click", () => whileBusy(downloadWeather));
element("weather-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (file) await whileBusy(async () => loadWeather(await file.text(), file.name, element("weather-source").value));
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
element("load-samples").addEventListener("click", () => whileBusy(async () => {
  const [weatherText, skyText] = await Promise.all([fetchText(SAMPLE_WEATHER.url), fetchText(SAMPLE_SKY.url)]);
  await loadWeather(weatherText, SAMPLE_WEATHER.url.split("/").pop(), SAMPLE_WEATHER.source);
  await loadObstructedSky(skyText, SAMPLE_SKY.url.split("/").pop());
}));
element("compute").addEventListener("click", () => whileBusy(compute, "computed"));
for (const id of ENTER_COMPUTES) {
  element(id).addEventListener("keydown", event => { if (event.key === "Enter" && !element("compute").disabled) element("compute").click(); });
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
element("forget").addEventListener("click", () => whileBusy(async () => {
  await storeRequest("readwrite", store => store.clear());
  report("Forgot the stored inputs; they stay loaded until the page is reloaded");
}));
element("export-zip").addEventListener("click", () => whileBusy(async () => saveFile("pv_yield_results.zip", await call("exportZip"), "application/zip"), "computed"));
element("export-pdf").addEventListener("click", () => whileBusy(async () => saveFile("pv_yield_report.pdf", await call("pdfReport"), "application/pdf"), "computed"));

try {
  updateButtons();
  const versions = await call("init");
  report(`Python ready: ${Object.entries(versions).map(([name, version]) => `${name} ${version}`).join(", ")}`);
  // BokehJS must match the Python bokeh version exactly.
  report(`Loading BokehJS ${versions.bokeh}`);
  await loadScript(`https://cdn.bokeh.org/bokeh/release/bokeh-${versions.bokeh}.min.js`);
  pythonReady = true;
  await whileBusy(restoreStoredInputs);
  if (document.body.dataset.state === "ready") report("Ready: set the site and load or download weather data, load an obstructed sky description (or the sample data), then compute");
} catch (error) {
  fail(error);
}
