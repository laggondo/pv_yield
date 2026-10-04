# Concept

## Purpose

Estimate the photovoltaic (PV) yield for a user-defined location:

- Use site weather data (from various possible sources).
- Use a description of the obstructed sky hemisphere, i.e. which parts of the sky are blocked by buildings, trees or terrain as seen from the site (derived with different possible methods).
- Consider both direct and diffuse radiation.
- Convert irradiation to yield with simple, user-defined metrics that have sensible defaults.
- Assess the yield with numeric key figures (KPIs) and visualizations: annual, seasonal/monthly, and a typical day per month.
- Support analysis and optimization of panel orientation.
- Be usable in a browser on both computers and phones.
- Possible future extension: a planning tool beyond PV, e.g. for home battery systems.

## Platforms

- Browser app for computers and phones.
- Hosted as a static site on GitHub (no server); all computation runs in the browser, written in Python via Pyodide (Python compiled to run in the browser).
- Pyodide was checked against alternatives (PyScript, a JavaScript rewrite, a hybrid) and confirmed (#6). Pages that don't need Python, e.g. marking obstructions in a photo, may be plain JavaScript.
- Parts of the method may use the phone's camera, GPS, accelerometer, gyroscope and compass.
- Parts of the method are well suited for touch screens.
- The pipeline runs in separate steps, not necessarily in one go. Intermediate results are stored on the device as files (JSON for data, YAML for configs) that can be transferred to other devices, e.g. measure on the phone, analyze on the computer. Offline use is not a priority, but stored results allow some steps to run offline.

## Inputs

- **Weather data:** e.g. an hourly typical meteorological year (TMY3 format), established via the paid MeteoNorm service (see `legacy_code/`), or from the free PVGIS service (sample in `data/`). Explore free alternatives of similar quality, ideally downloaded automatically.
  - Browsers block requests from a static page to other sites unless those sites allow it (CORS). PVGIS (the EU's free solar data service) does not; Open-Meteo (reanalysis, real years averaged into a typical year) does (#8). So the browser downloads from Open-Meteo and links to the PVGIS file for a manual download; the CLI prefers PVGIS and falls back to Open-Meteo.
- **Site selection:** place name or address (geocoding with Nominatim, OpenStreetMap), GPS on phones, or coordinates; the weather data comes from the nearest point the chosen service offers.
- **Site coordinates** (latitude/longitude); may come from the weather data or from user inputs.
- **Panel orientation** (tilt and azimuth), unless it is optimized.
- **Sky obstruction, LiDAR-based:** point cloud from a LiDAR scanner (e.g. Livox), see `legacy_code/`. Scans may be much larger than the 17 MB sample, so this step is meant for computers, not phones.
  - Scanner alignment relative to geographic north (yaw angle), entered manually.
  - Local panel offset: position of the panel relative to the scanner.
- **Sky obstruction, photo-based (alternative):** photos of the sky, plus metadata per photo: viewing direction (compass heading, and tilt from accelerometer/gyroscope), camera field of view, GPS position and timestamp.
- **PV panel parameters:** area, efficiency, performance ratio, etc.

Not considered for now: radiation reflected from the ground.

## Outputs

All integral outputs compare obstructed with unobstructed incident radiation. More outputs may be added later.

- **Numeric/tabular:**
  - Annual radiation and PV yield, with radiation split into direct and diffuse.
  - Average daily radiation and PV yield for each month.
  - Specific yield: annual yield per rated panel power (kWh/kWp, kilowatt peak), for comparison between sites.
  - Annual shading loss (%) and sky view factor (fraction of the diffuse radiation that reaches the panel despite obstructions), as in `legacy_code/` (which computes it for a horizontal surface only).
- **Visualizations:**
  - Discretized sky hemisphere with obstructions and the annual sun path, colored by radiation, so it shows how much radiation each blocked part of the sky costs. Independent of panel orientation.
  - Average daily profiles (obstructed and unobstructed) for each month.
  - Annual bar plot with one bar per day, obstructed and unobstructed radiation.
  - Carpet plot: hourly radiation over the year as a heatmap (day of year vs. hour of day), obstructed and unobstructed, as in `legacy_code/`.
- **Panel orientation:** easy comparison of results for different orientations, e.g. a heatmap of annual yield over tilt and azimuth with the best orientation marked.
- **Export:** results as CSV or JSON together with the config used (YAML), so runs can be compared later; a PDF report with the key figures and plots.

## Pipeline

Most of the functionality already exists in `legacy_code/`; it serves as a reference, not as code to refactor.

### Obstructed sky description

The obstructed sky description is the fundamental intermediate result:

- Uses a discretization of the sky hemisphere into triangular sky patches, each defined by three nodes. Most operations work on these patches.
- The discretization is fine enough that no operations below patch level are needed.
- Independent of sun position, weather data and panel orientation.
- Can be stored and re-used later or on a different device, in a human-readable file (JSON) whose format is the same for all methods:
  - Nodes as unit vectors, triangles as triples of node indices, and one obstructed yes/no flag per triangle. Storing the discretization itself (not just its parameters) keeps files readable if the default resolution changes.
  - Metadata such as location and date; the producing method and its settings are recorded for information only and don't change the format.
- Coordinate system: x points east, y north, z up. Azimuths follow the compass (0° = north, clockwise). Note that `legacy_code/` uses a different convention (Duffie-Beckman: azimuth 0° = south, x west, y south).
- Can be produced by different methods, listed below. The implementation makes it easy to add further methods. (In code, name the methods by what they do, e.g. LiDAR or photo, not by letters.)
- Evaluated at a single point per panel (e.g. its centre) for the first version (#10); area evaluation could later combine several obstructed sky descriptions, one per sample point on the panel.

**LiDAR method** (see `legacy_code/`):

- Input is a LiDAR point cloud.
- Points are normalized to unit vectors, giving directions from the scanner.
- A sky patch counts as obstructed if it contains at least a minimum number of points; this filters out noise. The minimum is configurable, either as an absolute count or as a percentage of all points.
- Accounts for the panel's offset from the scanner position.

**Photo method** (#17, in the browser app; plain JavaScript for drawing and touch input, the camera geometry in Python, `core/photo.py`):

- Input is a photo of the relevant part of the sky, combined with sensor metadata on camera orientation and field of view.
  - Photos are taken in the page with the phone's camera; the camera orientation (viewing azimuth, elevation, roll) comes from the absolute device orientation (compass, accelerometer, gyroscope) at the moment of the shot. Existing photos can be loaded, with the orientation entered by hand.
  - Browsers don't report the camera's field of view; it is an input (default 65° across the longer image side), checked on a photo.
  - The orientation of each photo can be corrected so that the horizon and the compass letters drawn over the photo match it (sensor errors, magnetic declination): by moving the sky grid (right mouse button or two fingers; rotating, and zooming for the field of view) or by entering the angles.
- The sky discretization is overlaid on the photo with a pinhole camera model, which maps the patch edges (great circle arcs) to straight lines, and the user marks obstructed sky patches manually (touchscreen or mouse): a tap toggles a patch, a drag marks or frees all patches it passes over.
- The same marking works on a map of the sky hemisphere (north up, as in the sky plot), without a photo.
- Several photos can be combined: all views edit the same flags. The sky map shows the photos merged onto the hemisphere (where photos overlap, each direction is taken from the photo that sees it closest to its image centre, the least distorted), and while taking a photo, the patches covered by earlier photos are highlighted. Sky patches not covered by any photo keep their start state (free, unless a loaded description marks them).
- The photos can be saved as a photo set (JSON with the images and their camera views) to continue on another device.
- No offset between camera and panel (#11): the photo is taken from the panel position, and the page says so. Estimating distances (entered per obstruction, or from two photos) may follow later.
- Automatic sky detection may follow later.

**Combining methods** (#12): a loaded or computed obstructed sky description (e.g. from LiDAR) is the start state of the manual marking, so a LiDAR result can be corrected on photos or on the sky map. The edits are baked into the flags (no separate layer; the file format stays the same). The metadata keeps the original entries and records the contributing methods in order (`methods`, e.g. `[lidar, photo]`, joined as `method: lidar+photo`) and one entry per editing session under `edits` (date, methods, number of changed patches, the photos' camera views).

### Irradiation and yield

Brings in the sun position over the year and the weather data, and produces the outputs; mostly extracted from `legacy_code/`.

- Weather data from multiple sources; at least one option downloads automatically from a free source.
  - A typical meteorological year (TMY) is preferred; for sources with real historical years, average several years.
  - Sources may differ in time resolution (e.g. 15-minute, hourly, or only monthly values). Internally, weather data is hourly; each source converts its data accordingly (e.g. finer data is summed up to hours).
  - *Open question:* can "generic" hourly irradiation time series be produced from coarse data, e.g. only monthly values, that are still sufficient to assess the PV yield?
  - Each weather source handles its own time stamp convention (e.g. TMY3: local standard time, value covers the hour before its time stamp; Open-Meteo: UTC), so sun positions match the data exactly. Otherwise morning and evening shading shifts by up to an hour.
- Intermediate result: the irradiation assigned to each sky patch, per hour. Like the obstructed sky description, it can be stored and re-used. It is independent of the obstruction and the panel orientation, so both can be varied without recomputing it.
- With a fine discretization, hourly sun positions skip sky patches entirely: the sun moves up to ~15° per hour, more than a patch width. Therefore each hour is split into sub-steps small enough that the sun moves less than about half a patch per step (the legacy code uses sub-steps too, via `n_sub_steps`), and the hour's direct radiation is distributed over the patches of these sub-step sun positions.
- Direct radiation per sky patch and hour is stored as a vector (three numbers), not a single value, so that radiation on the panel can be computed exactly for any orientation later:
  - The direct radiation a panel receives from one sun position is the direct normal radiation E times cos(θ), where θ is the angle between the sun direction and the panel normal. With unit vectors s (towards the sun) and n (panel normal), cos(θ) = s · n (dot product), so the panel receives E · (s · n).
  - Within one hour, the sun passes through a patch at several sub-steps k, each with its own direction s_k and radiation E_k. The panel receives Σ E_k (s_k · n) = (Σ E_k s_k) · n.
  - So storing the single vector V = Σ E_k s_k per patch and hour is enough: for any panel orientation, the direct radiation from that patch is V · n, one dot product (zero if negative, i.e. the patch lies behind the panel). This avoids approximating all sun positions by the patch center.
  - The length of V is the direct normal radiation from the patch, up to a negligible error since sun directions within one patch differ by only a few degrees; this orientation-independent value is used e.g. for the sky plot.
- Diffuse radiation is distributed evenly over the sky (isotropic) and weighted per patch by its angle to the panel normal. This fixes the legacy approach, which uses horizontal diffuse radiation unchanged for any tilt. Brighter zones near the horizon and around the sun are ignored for now.
- Obtaining the outputs from the irradiation per sky patch, the obstructed sky description and a panel orientation:
  - Hourly direct and diffuse radiation on the panel: sums over the sky patches, excluding obstructed patches (obstructed) or not (unobstructed).
  - Daily, monthly and annual values and the carpet plot: sums of the hourly values.
  - PV yield: radiation on the panel × area × efficiency × performance ratio. Specific yield (kWh/kWp) is radiation on the panel × performance ratio, independent of area and efficiency.
  - Shading loss: 1 − obstructed / unobstructed radiation.
  - Orientation comparison: needs only annual sums per sky patch (the vectors V can be summed over the year as well), so a fine grid of orientations is fast to compute.

## Technical decisions

- Object-oriented implementation. Where several methods produce the same result (sky obstruction, weather data source), they share a common interface, so further methods can be added easily.
- Reimplement rather than import modules from `legacy_code/`.
- First version: only the LiDAR method for the obstructed sky description; the photo method followed in bundle 5.
- One internal config dict holds all user choices and parameters; it can be exported to and imported from YAML.
- Core functionality and algorithms live in a library (a Python package within this repository), used by all front ends: the browser app and a command-line app (CLI) for the computer, which suits the LiDAR processing. The core does no user interaction and no file access: it takes data and returns results. User interface and file handling stay in the front ends; plotting is a separate part of the library, so both front ends share the same plots.
- Browser front end: plain HTML/JavaScript for the user interface, with the Python library running underneath via Pyodide. This gives full control over touch input, camera and sensors, which the photo method needs. Python runs in a Web Worker so the page stays responsive, the Pyodide version is pinned, and each page loads only the packages it needs. The first-visit download (~30 MB for pvlib on top of Pyodide, then cached by the browser) is acceptable.
- Sun position and other solar calculations: pvlib (BSD-3-Clause license, permissive). Verified: pvlib 0.16.1 installs and computes sun positions in Pyodide 314.0.7 (Python 3.14); numpy, scipy, pandas, matplotlib and Bokeh are bundled with Pyodide.
- File formats: configs (exported or hand-written) are YAML; JSON only for larger data files such as the obstructed sky description and the irradiation per sky patch.
- Every stored file (config, obstructed sky description, irradiation per sky patch) contains a format version, so older files stay readable after format changes.
- Process whole arrays with numpy instead of Python loops over points or hours (the legacy code loops), especially since Python runs slower in the browser.
- Repository layout: the package in `src/`, with `pyproject.toml` and a conda `environment.yml` (as in `satellite-heliostat-extractor`).
- Plotting: interactive plots with Bokeh (bundled with Pyodide; chosen over Plotly, Vega-Lite/Altair, ECharts and uPlot, #7), static plots with matplotlib for PDF reports. The same Python plotting code serves both front ends: the CLI writes standalone HTML, the browser embeds the plot as JSON (`json_item`) with BokehJS matching the Python bokeh version exactly.
