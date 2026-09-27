# Pyodide proof page

Minimal check for the platform (#6) and plotting (#7) decisions: loads Pyodide, installs pvlib with micropip, computes a year of sun positions in 10-minute steps and shows a sky hemisphere plot with Bokeh, plus load timings.

Run from the repository root and open <http://localhost:8000/web/proof/> (the first load downloads several tens of MB, mostly pvlib with its 19 MB wheel, scipy, pandas and the Pyodide core; later loads come from the browser cache):

```
python -m http.server 8000
```

- `?pyodide=<base URL>` loads Pyodide from elsewhere, e.g. a self-hosted copy of the full distribution.
- `python web/proof/proof.py sky_plot.html` builds the same plot natively (conda environment plus `bokeh`) as standalone HTML, as the CLI would.
