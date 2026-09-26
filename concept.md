# Concept

## Purpose

Estimate the photovoltaic (PV) yield for a user-defined location:

- Use site weather data (from various possible sources).
- Use a description of the unobstructed sky hemisphere, i.e. which parts of the sky are visible from the site and not blocked by buildings, trees or terrain (derived with different possible methods).
- Consider both direct and diffuse radiation.
- Convert irradiation to yield with simple, user-defined metrics that have sensible defaults.
- Assess the yield with numeric key figures (KPIs) and visualizations: annual, seasonal/monthly, and a typical day per month.
- Support analysis and optimization of panel orientation.
- Be usable in a browser on both computers and phones.
- Possible future extension: a planning tool beyond PV, e.g. for home battery systems.

## Platforms

- Browser app for computers and phones.
- Hosted as a static site on GitHub (no server); all computation runs in the browser, written in Python (e.g. via Pyodide, Python compiled to run in the browser).
- Parts of the method may use the phone's camera, GPS, accelerometer, gyroscope and compass.
- Parts of the method are well suited for touch screens.
- The pipeline runs in separate steps, not necessarily in one go. Intermediate results are stored on the device as files (e.g. JSON) that can be transferred to other devices, e.g. measure on the phone, analyze on the computer. Offline use is not a priority, but stored results allow some steps to run offline.

## Inputs

- **Weather data:** hourly typical meteorological year (TMY3 format), established via the paid MeteoNorm service (see `legacy_code/`, sample in `data/`). Explore free alternatives of similar quality, ideally downloaded automatically.
- **Site coordinates** (latitude/longitude); may come from the weather data.
- **Panel orientation** (tilt and azimuth), unless it is optimized.
- **Sky obstruction, LiDAR-based:** point cloud from a LiDAR scanner (e.g. Livox), see `legacy_code/`.
  - Scanner alignment relative to geographic north (yaw angle); set by hand in the legacy code.
  - Local panel offset: position of the panel relative to the scanner.
- **Sky obstruction, photo-based (alternative):** photos of the sky, plus metadata per photo: viewing direction (compass heading, and tilt from accelerometer/gyroscope), camera field of view, GPS position and timestamp.
- **PV panel parameters:** area, efficiency, performance ratio, etc.

## Outputs

## Pipeline

## Technical decisions
