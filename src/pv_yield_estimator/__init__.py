"""PV yield estimator: yield of a photovoltaic panel from weather data and an obstructed sky description.

Subpackages:
- `core`: algorithms; no user interaction and no file access, takes data and returns results.
- `plotting`: plots shared by the CLI and the browser front end.
- `cli`: command-line front end.
- `browser`: Python side of the browser front end, called from the Pyodide worker; the static HTML/JavaScript part
  lives in `web/` at the repository root.
"""

__version__ = "0.1.0"
