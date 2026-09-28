# Sample data

Reference samples for development and tests (see `CLAUDE.md`).

| File | Content | Source |
|---|---|---|
| `2026-06-01_22-25-29_red_red.csv` | LiDAR point cloud (Livox Mid-360 CSV export) | scan of the sample site |
| `Freiburg-hour.csv` | Typical meteorological year (TMY3 format), Freiburg | MeteoNorm (paid service) |
| `Freiburg-pvgis-tmy.csv` | Typical meteorological year, Freiburg (48.000° N, 7.850° E) | PVGIS, see below |

## PVGIS typical meteorological year

- Source: PVGIS (Photovoltaic Geographical Information System) © European Union, 2001–2026, <https://re.jrc.ec.europa.eu/pvg_tools/>. Free to reuse with this attribution.
- Downloaded on 2026-09-28 from the PVGIS API, version 5.3, default database and period: `https://re.jrc.ec.europa.eu/api/v5_3/tmy?lat=48.0&lon=7.85&outputformat=csv`
- Hourly values in UTC; each month is taken from a different real year (listed in the file header). Columns: `G(h)` global horizontal, `Gb(n)` direct normal and `Gd(h)` diffuse horizontal irradiance in W/m², plus temperature, humidity, wind and pressure. The format differs from TMY3, so the program can't read it yet; a PVGIS reader is planned with the weather sources in bundle 4 (#8, #19).
- Annual sums: global horizontal 1178 kWh/m², direct normal 1135 kWh/m², diffuse horizontal 587 kWh/m² (MeteoNorm file: 1147, 1050, 597).
