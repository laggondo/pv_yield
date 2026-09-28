# Sample data

Reference samples for development and tests (see `CLAUDE.md`).

| File | Content | Source |
|---|---|---|
| `2026-06-01_22-25-29_red_red.csv` | LiDAR point cloud (Livox Mid-360 CSV export) | own measurement by the repository owner |
| `Freiburg-pvgis-tmy.csv` | Typical meteorological year, Freiburg (48.000° N, 7.850° E) | PVGIS, see below |

## PVGIS typical meteorological year

- Source: PVGIS (Photovoltaic Geographical Information System) © European Union, 2001–2026, <https://re.jrc.ec.europa.eu/pvg_tools/>. Free to reuse with this attribution.
- Downloaded on 2026-09-28 from the PVGIS API, version 5.3, default database and period: `https://re.jrc.ec.europa.eu/api/v5_3/tmy?lat=48.0&lon=7.85&outputformat=csv`
- Hourly values in UTC; each month is taken from a different real year (listed in the file header). Columns: `G(h)` global horizontal, `Gb(n)` direct normal and `Gd(h)` diffuse horizontal irradiance in W/m², plus temperature, humidity, wind and pressure. Read with `weather.source: pvgis_tmy` in the config (the default of `examples/sample_config.yaml`). Each value is the satellite estimate at the time stamp plus the "irradiance time offset" in the header (0.18 h) and is taken as the mean of the hour centred there.
- Annual sums: global horizontal 1178 kWh/m², direct normal 1135 kWh/m², diffuse horizontal 587 kWh/m².
