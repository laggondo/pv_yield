"""Build the static site of the browser app: the files in web/ and the package as a zip for Pyodide.

Standard library only, so the GitHub Pages workflow needs no environment. Local use, from the repository root:
    python web/build_site.py && python -m http.server 8000 -d _site
then open http://localhost:8000/. The site's settings (`site_settings.js`) come from the arguments, e.g. the Google Maps
API key for the panel map (default: the environment variable GOOGLE_MAPS_API_KEY; empty: Esri imagery instead).
"""

import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
WEB_DIRECTORY = REPOSITORY / "web"
PACKAGE_DIRECTORY = REPOSITORY / "src" / "pv_yield_estimator"


def zip_package(package_directory, zip_path):
    """Zip the package's Python files with the package directory at the top level, as Pyodide unpacks it into site-packages."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_directory.rglob("*.py")):
            archive.write(path, path.relative_to(package_directory.parent).as_posix())


def build_site(output_directory, google_maps_api_key=""):
    """Write the site into `output_directory`, replacing its previous content, with the given settings; returns the directory."""
    output_directory = Path(output_directory)
    if output_directory.exists():
        shutil.rmtree(output_directory)
    shutil.copytree(WEB_DIRECTORY, output_directory, ignore=shutil.ignore_patterns("build_site.py", "__pycache__", "*.pyc"))
    zip_package(PACKAGE_DIRECTORY, output_directory / "pv_yield_estimator.zip")
    settings = (output_directory / "site_settings.js").read_text(encoding="utf-8")
    settings = re.sub(r'GOOGLE_MAPS_API_KEY = ".*";', f"GOOGLE_MAPS_API_KEY = {json.dumps(google_maps_api_key)};", settings)
    (output_directory / "site_settings.js").write_text(settings, encoding="utf-8")
    return output_directory


def main():
    """Parse the arguments and build the site."""
    parser = argparse.ArgumentParser(description="Build the static site of the browser app.")
    parser.add_argument("-o", "--output", type=Path, default=REPOSITORY / "_site", help="output directory (default: _site in the repository root); its content is replaced")
    parser.add_argument("--google-maps-api-key", default=os.environ.get("GOOGLE_MAPS_API_KEY", ""), help="Google Maps JavaScript API key for the panel map (default: the environment variable GOOGLE_MAPS_API_KEY); empty: Esri imagery")
    args = parser.parse_args()
    output_directory = build_site(args.output, args.google_maps_api_key)
    print(f"Built the site in {output_directory}; serve it e.g. with: python -m http.server 8000 -d {output_directory}")


if __name__ == "__main__":
    main()
