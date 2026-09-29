// User interface of the browser app; the computation runs in Python in a Web Worker (worker.js).
// The page state is in document.body.dataset.state: loading, ready, busy, computed or error (used by the smoke test and
// the red/green status light); progress messages and timings go to the log at the bottom.

const SAMPLE_WEATHER = { url: "samples/Freiburg-pvgis-tmy.csv", source: "pvgis_tmy" };
const SAMPLE_SKY = { url: "samples/sample_obstructed_sky.json" };
const PANEL_ENTRIES = ["tilt_deg", "azimuth_deg", "area_m2", "efficiency", "performance_ratio"];
const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const pageStart = performance.now();
const element = id => document.getElementById(id);
const worker = new Worker(`worker.js${location.search}`, { type: "module" });
const pendingCalls = new Map();
let nextCallId = 0;
let pythonReady = false;
let weather = null;          // {text, filename}; kept to re-parse when the format changes
const loaded = { weather: false, sky: false };
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
  const busy = document.body.dataset.state === "busy";
  element("load-samples").disabled = !pythonReady || busy;
  element("compute").disabled = !pythonReady || busy || !loaded.weather || !loaded.sky;
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

const format = (value, digits) => Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const percent = value => `${format(100 * value, 1)} %`;

// Guess the weather file format from its first line: PVGIS files start with "Latitude (decimal degrees)".
function guessWeatherSource(text) {
  return text.startsWith("Latitude") ? "pvgis_tmy" : "tmy3";
}

async function loadWeather(text, filename, source) {
  weather = { text, filename };
  loaded.weather = false;
  element("weather-source").value = source;
  element("weather-summary").textContent = "loading ...";
  const summary = await call("loadWeather", text, filename, source);
  loaded.weather = true;
  element("weather-summary").textContent = `${filename}: ${summary.name}, ${format(summary.latitude, 3)}° N, ${format(summary.longitude, 3)}° E, ${format(summary.altitude, 0)} m; ${summary.n_hours} hours; annual GHI ${format(summary.annual_ghi_kwh_m2, 0)}, DNI ${format(summary.annual_dni_kwh_m2, 0)}, DHI ${format(summary.annual_dhi_kwh_m2, 0)} kWh/m²`;
  report(`Weather loaded: ${filename}`);
}

async function loadObstructedSky(text, filename) {
  loaded.sky = false;
  element("sky-summary").textContent = "loading ...";
  const summary = await call("loadObstructedSky", text, filename);
  loaded.sky = true;
  element("sky-summary").textContent = `${filename}: ${summary.n_obstructed} of ${summary.n_patches} sky patches obstructed (${percent(summary.obstructed_solid_angle_fraction)} of the solid angle), horizontal sky view factor ${format(summary.sky_view_factor_horizontal, 3)}; method ${summary.method || "?"}${summary.input_file ? ` from ${summary.input_file}` : ""}`;
  report(`Obstructed sky description loaded: ${filename}`);
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch ${url}: HTTP ${response.status}`);
  return response.text();
}

// The config as a partial config dict; missing entries take the defaults of the Python functions.
function currentConfig() {
  const panel = {};
  for (const name of PANEL_ENTRIES) {
    const value = element(name).valueAsNumber;
    if (Number.isNaN(value)) throw new Error(`Panel entry ${name} is not a number: ${JSON.stringify(element(name).value)}`);
    panel[name] = value;
  }
  return { weather: { source: element("weather-source").value }, panel };
}

function showKeyFigures(figures) {
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
  element("key-figures").innerHTML = `<tr><th>Annual</th><th class="number">unobstructed</th><th class="number">obstructed</th></tr>`
    + pairs.map(([label, prefix, suffix, digits]) => `<tr><td>${label}</td><td class="number">${format(figures[`${prefix}_unobstructed${suffix}`], digits)}</td><td class="number">${format(figures[`${prefix}_obstructed${suffix}`], digits)}</td></tr>`).join("")
    + singles.map(([label, value]) => `<tr><td>${label}</td><td></td><td class="number">${value}</td></tr>`).join("");
}

function showMonthly(rows) {
  element("monthly").innerHTML = `<tr><th>Month</th><th class="number">radiation unobstructed (kWh/m²/d)</th><th class="number">radiation obstructed (kWh/m²/d)</th><th class="number">yield unobstructed (kWh/d)</th><th class="number">yield obstructed (kWh/d)</th></tr>`
    + rows.map(row => `<tr><td>${MONTH_NAMES[row.month - 1]}</td><td class="number">${format(row.total_unobstructed, 2)}</td><td class="number">${format(row.total_obstructed, 2)}</td><td class="number">${format(row.yield_unobstructed, 3)}</td><td class="number">${format(row.yield_obstructed, 3)}</td></tr>`).join("");
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
  report(`Computing: tilt ${config.panel.tilt_deg}°, azimuth ${config.panel.azimuth_deg}° ...`);
  const result = await call("compute", config);
  window.pvYieldApp.lastResult = result;
  element("results").hidden = false;
  showKeyFigures(result.key_figures);
  showMonthly(result.monthly_daily_average);
  await showPlots(result.plots);
  report(`Computed (${Object.entries(result.timings).map(([step, seconds]) => `${step} ${seconds.toFixed(1)} s`).join(", ")})`);
}

element("weather-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  await whileBusy(() => loadWeather(text, file.name, guessWeatherSource(text)));
});
element("weather-source").addEventListener("change", async event => {
  if (weather) await whileBusy(() => loadWeather(weather.text, weather.filename, event.target.value));
});
element("sky-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (file) await whileBusy(async () => loadObstructedSky(await file.text(), file.name));
});
element("load-samples").addEventListener("click", () => whileBusy(async () => {
  const [weatherText, skyText] = await Promise.all([fetchText(SAMPLE_WEATHER.url), fetchText(SAMPLE_SKY.url)]);
  await loadWeather(weatherText, SAMPLE_WEATHER.url.split("/").pop(), SAMPLE_WEATHER.source);
  await loadObstructedSky(skyText, SAMPLE_SKY.url.split("/").pop());
}));
element("compute").addEventListener("click", () => whileBusy(compute, "computed"));
for (const name of PANEL_ENTRIES) {
  element(name).addEventListener("keydown", event => { if (event.key === "Enter" && !element("compute").disabled) element("compute").click(); });
}

try {
  const versions = await call("init");
  report(`Python ready: ${Object.entries(versions).map(([name, version]) => `${name} ${version}`).join(", ")}`);
  // BokehJS must match the Python bokeh version exactly.
  report(`Loading BokehJS ${versions.bokeh}`);
  await loadScript(`https://cdn.bokeh.org/bokeh/release/bokeh-${versions.bokeh}.min.js`);
  pythonReady = true;
  setState("ready");
  report("Ready: load a weather file and an obstructed sky description, or the sample data");
  updateButtons();
} catch (error) {
  fail(error);
}
