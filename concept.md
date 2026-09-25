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

## Outputs

## Pipeline

## Technical decisions
