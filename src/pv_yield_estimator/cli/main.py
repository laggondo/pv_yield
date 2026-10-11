"""Entry point of the command-line front end.

Subcommands run the pipeline steps, each reading and writing intermediate files:
- `obstruction`: LiDAR point cloud → obstructed sky description (JSON); AR recording → reconstructed point cloud (CSV, #48)
- `weather`: download weather data for the site (site.latitude/longitude or site.query) → weather file
- `irradiation`: weather file → irradiation per sky patch (JSON)
- `yield`: irradiation per sky patch + obstructed sky description (optional) → results directory (key figures, tables, plots, report)
- `run`: all steps in one go: point cloud or obstructed sky description (optional) + weather file (downloaded if not given) → results directory
Without an obstructed sky description, the sky is free (no obstruction).
Without a subcommand, the config is only assembled, logged and optionally exported.
With `-p/--project` (a project zip as saved by the browser app, or a directory with the same layout), the project's
config comes first, inputs not given on the command line are taken from the project, and the inputs given and the
outputs are written into it (with the config used), so the project can go back to the browser.
All file and network access of the CLI lives here; the core receives and returns data.
"""

import argparse
import json
import logging
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from bokeh.io import save
from bokeh.layouts import column
from bokeh.resources import INLINE

from pv_yield_estimator import __version__
from pv_yield_estimator.config import apply_overrides, config_from_yaml, config_to_yaml, default_config, log_config, merge_configs, parse_overrides
from pv_yield_estimator.core.export import export_files
from pv_yield_estimator.core.irradiation import PatchIrradiation, compute_patch_irradiation
from pv_yield_estimator.core.obstruction_methods import sky_obstruction_method
from pv_yield_estimator.core.panel_yield import YieldEstimator
from pv_yield_estimator.core.ar_recording import check_recording
from pv_yield_estimator.core.sky import ObstructedSky, SkyDiscretization
from pv_yield_estimator.core.weather import distance_km, load_weather, resolve_site
from pv_yield_estimator.core.weather_download import USER_AGENT, parse_site_search, site_search_url, weather_download_candidates
from pv_yield_estimator.file_format import to_json_text
from pv_yield_estimator.logging_setup import setup_logging
from pv_yield_estimator.plotting.interactive import result_plots
from pv_yield_estimator.plotting.report import pdf_report as make_pdf_report
from pv_yield_estimator.plotting.static import reconstruction_figure, static_figures
from pv_yield_estimator.project import (CONFIG_NAME, IRRADIATION_NAME, OBSTRUCTED_SKY_NAME, POINT_CLOUD_STEM, RECONSTRUCTED_POINTS_NAME, RECORDING_NAME, REPORT_NAME, RESULTS_DIRECTORY, WEATHER_STEM,
                                        ProjectFiles)

log = logging.getLogger(__name__)


def add_common_arguments(parser, defaults=True):
    """Add the config and logging options; subcommands get them without defaults, so they only override if given."""
    suppress = {} if defaults else {"default": argparse.SUPPRESS}
    parser.add_argument("-c", "--config", action="append", type=Path, **({"default": []} | suppress), help="YAML config file; may be repeated, later files override earlier ones")
    parser.add_argument("-m", "--mod", nargs="+", action="extend", metavar="KEY=VALUE", **({"default": []} | suppress), help="override config entries, e.g. -m panel.tilt_deg=30 site.latitude=48.0; values are parsed as YAML")
    parser.add_argument("--export-config", type=Path, metavar="FILE", **({"default": None} | suppress), help="write the assembled config to this YAML file")
    parser.add_argument("-l", "--log-level", **({"default": "info"} | suppress), help="log level: debug, info, warning, error (default: info)")
    parser.add_argument("-p", "--project", type=Path, metavar="PROJECT", **({"default": None} | suppress),
                        help="project zip (as saved by the browser app) or directory with the same layout: its config comes first, missing inputs are taken from it, inputs and outputs are written into it (created if new)")


def parse_arguments(argv=None):
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(prog="pv-yield-estimator", description="Estimate photovoltaic yield.")
    add_common_arguments(parser)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    obstruction = subparsers.add_parser("obstruction", help="LiDAR point cloud -> obstructed sky description (JSON); AR recording -> reconstructed point cloud (CSV)")
    obstruction.add_argument("input", type=Path, nargs="?", help="point cloud file (e.g. Livox CSV), or an AR recording (recording.json, or the directory with it and its frames); without it, from the project")
    obstruction.add_argument("-o", "--output", type=Path, help="file to write: obstructed sky description (JSON), or for an AR recording the reconstructed point cloud (CSV); optional with -p")
    obstruction.add_argument("--plot", type=Path, metavar="PNG", help="AR recording: plot of the reconstructed point cloud from above and from the south, with the camera path")

    weather = subparsers.add_parser("weather", help="download weather data for the site (site.latitude/longitude or site.query) -> weather file")
    weather.add_argument("-o", "--output", type=Path, help="weather file to write (default: a name with service and site in the current directory)")

    irradiation = subparsers.add_parser("irradiation", help="weather file -> irradiation per sky patch (JSON)")
    irradiation.add_argument("weather", type=Path, nargs="?", help="weather file: PVGIS TMY (CSV/JSON), Open-Meteo (JSON) or TMY3 (CSV); the format is detected; without it, from the project")
    irradiation.add_argument("-o", "--output", type=Path, help="irradiation per sky patch to write (JSON); optional with -p")
    irradiation.add_argument("--sky", type=Path, help="obstructed sky description whose sky discretization to use (default: from simulation.n_sky_nodes)")

    yield_parser = subparsers.add_parser("yield", help="irradiation per sky patch + obstructed sky description (optional) -> results directory")
    yield_parser.add_argument("irradiation", type=Path, nargs="?", help="irradiation per sky patch (JSON); without it, from the project")
    yield_parser.add_argument("obstructed_sky", type=Path, nargs="?", help="obstructed sky description (JSON); without it, from the project, else no obstruction")
    yield_parser.add_argument("-d", "--output-dir", type=Path, help="directory for the results (see the config's output section); without it, the key figures are only logged")

    run = subparsers.add_parser("run", help="all steps: point cloud or obstructed sky description (optional) + weather file (downloaded if not given) -> results directory")
    run.add_argument("-s", "--sky-input", type=Path, help="point cloud file (e.g. Livox CSV), or an obstructed sky description (JSON) to skip the obstruction step; without it, from the project, else no obstruction")
    run.add_argument("-w", "--weather", type=Path, help="weather file; if not given, from the project, else downloaded for the site (site.latitude/longitude or site.query) and kept in the output directory")
    run.add_argument("-d", "--output-dir", type=Path, help="directory for the intermediate files (obstructed_sky.json, patch_irradiation.json, the downloaded weather) and the results; optional with -p")

    for subparser in subparsers.choices.values():
        add_common_arguments(subparser, defaults=False)
    return parser.parse_args(argv)


def assemble_config(config_paths=(), override_items=(), project=None):
    """Load the project's config (if any) and the config files in order, merge them over the default config and apply the overrides."""
    config = default_config()
    if project is not None and project.has(CONFIG_NAME):
        config = merge_configs(config, config_from_yaml(project.read_text(CONFIG_NAME), source=f"{project.path}/{CONFIG_NAME}"))
    for path in config_paths:
        config = merge_configs(config, config_from_yaml(require_file(path, "config file").read_text(encoding="utf-8"), source=str(path)))
    return apply_overrides(config, parse_overrides(override_items))


def require_file(path, kind, hint=""):
    """Return the path if the file exists, else raise an error naming the kind of file and how to produce it."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"The {kind} {str(path)!r} does not exist{'; ' + hint if hint else ''}")
    return Path(path)


def write_text(path, text):
    """Write a text file, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log.info(f"Wrote {path}")


def read_json(path, kind, hint=""):
    """Read a JSON file into a dict."""
    return json.loads(require_file(path, kind, hint).read_text(encoding="utf-8"))


def fetch_text(url, timeout=180):
    """Download a URL as text, identifying the program; HTTP errors carry the service's message."""
    log.info(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} from {url}: {error.read().decode('utf-8', errors='replace')[:500]}") from error


def resolve_site_query(config, limit=5):
    """If the site section has a `query` (place name or address) but no coordinates, look it up and set latitude, longitude and name; returns the config."""
    site = config["site"]
    if not site.get("query") or (site.get("latitude") is not None and site.get("longitude") is not None):
        return config
    places = parse_site_search(fetch_text(site_search_url(site["query"], limit)))
    if not places:
        raise ValueError(f"Site search found no place for {site['query']!r}; try another spelling, or set site.latitude and site.longitude")
    place = places[0]
    log.info(f"Site search for {site['query']!r}: using {place['name']} at {place['latitude']:.5f}°, {place['longitude']:.5f}°" + (f" ({len(places) - 1} further matches: {'; '.join(other['name'] for other in places[1:])})" if len(places) > 1 else ""))
    site.update(latitude=place["latitude"], longitude=place["longitude"], name=site.get("name") or place["name"])
    return config


def download_weather(config, output=None, output_dir=Path("."), reuse=False):
    """Download weather data for the site from the first service that delivers (see `weather_download_candidates`) and write it.

    The file goes to `output`, or into `output_dir` under the service's file name; with `reuse`, an existing file of
    that name is used instead of downloading again (cache). Returns the path of the weather file.
    """
    site = config["site"]
    errors = []
    candidates = weather_download_candidates(**site, **config["weather"])
    for candidate in candidates:
        path = Path(output) if output is not None else Path(output_dir) / candidate["filename"]
        if reuse and path.is_file():
            log.info(f"Using the weather file downloaded earlier: {path}")
            return path
        try:
            content = fetch_text(candidate["url"])
            weather = load_weather(content, source=candidate["weather_source"])
        except (RuntimeError, urllib.error.URLError, ValueError) as error:
            log.warning(f"Weather download from {candidate['service']} failed: {error}")
            errors.append(f"{candidate['service']}: {error}")
            continue
        log.info(f"Weather from {candidate['description']}: data point {weather.latitude:.4f}°, {weather.longitude:.4f}° is {distance_km(site['latitude'], site['longitude'], weather.latitude, weather.longitude):.1f} km from the site")
        write_text(path, content)
        return path
    raise RuntimeError("No weather service delivered data for the site:\n" + "\n".join(errors) + "\nDownload a file manually (e.g. " + candidates[0]["url"] + ") and pass it as the weather file")


def run_obstruction(point_cloud_path, config, content=None):
    """Obstructed sky description from a point cloud file (or its `content` as text, e.g. from a project), with the configured method and sky discretization."""
    method = sky_obstruction_method(**config["sky_obstruction"])
    if content is None:
        with open(require_file(point_cloud_path, "point cloud file"), encoding="utf-8") as point_cloud_file:
            data = method.read_input(point_cloud_file)
    else:
        data = method.read_input(content)
    sky = SkyDiscretization.from_node_count(**config["simulation"])
    site = {key: config["site"][key] for key in ("latitude", "longitude", "altitude", "name") if config["site"].get(key) is not None}
    return method.obstructed_sky(sky, data, metadata={"input_file": Path(point_cloud_path).name, **site})


def run_irradiation(weather_path, config, sky=None, content=None, source=None):
    """Irradiation per sky patch from a weather file (or its `content` as text, e.g. from a project, with the weather `source` stored there unless the config sets one); sky discretization from the config unless given."""
    if content is None:
        content = require_file(weather_path, "weather file", "download one with the `weather` subcommand").read_text(encoding="utf-8")
    weather_settings = {"source": source, **config["weather"]} if source else config["weather"]
    weather = load_weather(content, **weather_settings)
    site = resolve_site(weather, **config["site"])
    if sky is None:
        sky = SkyDiscretization.from_node_count(**config["simulation"])
    irradiation = compute_patch_irradiation(weather, sky, **site, **config["simulation"])
    irradiation.metadata["input_file"] = Path(weather_path).name
    return irradiation


def is_recording_path(path):
    """Whether a path names an AR recording: a `recording.json` or a directory with one."""
    return Path(path).is_dir() or Path(path).name == RECORDING_NAME


def read_recording(path=None, project=None):
    """An AR recording (dict of `recording.json`) and its frames' JPEG bytes, from `recording.json` or the directory with it (frame paths relative to it), else from the project."""
    if path is not None:
        path = Path(path)
        json_path = path / RECORDING_NAME if path.is_dir() else path
        recording = check_recording(read_json(json_path, "AR recording", "record one in the browser app and save the project"), str(json_path))
        return recording, [require_file(json_path.parent / frame["file"], "frame image").read_bytes() for frame in recording["frames"]]
    recording = check_recording(json.loads(project.read_text(RECORDING_NAME)), f"{project.path}/{RECORDING_NAME}")
    return recording, [project.read_bytes(frame["file"]) for frame in recording["frames"]]


def run_reconstruction(recording, images, config):
    """Point cloud from an AR recording, with the settings of the config's `sky_obstruction.ar_video` section."""
    from pv_yield_estimator.core.reconstruction import reconstruct_points
    return reconstruct_points(recording, images, **(config["sky_obstruction"].get("ar_video") or {}))


def run_yield(irradiation, obstructed_sky, config, output_dir=None):
    """Compute the results, log the key figures, monthly values and orientation comparison, and write the results directory if given."""
    result = YieldEstimator(irradiation, obstructed_sky, **config).run()
    key_figures = result.key_figures()
    monthly = result.monthly_daily_average()
    log.info("Key figures (radiation in kWh/m², yield in kWh, specific yield in kWh/kWp):\n" + yaml.safe_dump({key: round(value, 4) for key, value in key_figures.items()}, sort_keys=False))
    summary = monthly[["total_unobstructed", "total_obstructed", "yield_unobstructed", "yield_obstructed"]].round(3)
    summary.columns = ["rad_unobs", "rad_obs", "yield_unobs", "yield_obs"]
    log.info(f"Average daily values per month: radiation on the panel (kWh/m²/d) and yield (kWh/d), unobstructed and obstructed:\n{summary.to_string()}")
    comparison = "\n".join(f"  {row['label']:18} tilt {row['tilt_deg']:4g}°, azimuth {row['azimuth_deg']:5g}°: {row['annual_total_obstructed_kwh_m2']:7.1f} kWh/m² obstructed, {row['annual_total_unobstructed_kwh_m2']:7.1f} unobstructed" for row in result.orientation_comparison)
    log.info(f"Orientation comparison (annual radiation on the panel):\n{comparison}")
    if output_dir is not None:
        site = {"latitude": irradiation.latitude, "longitude": irradiation.longitude, "altitude": irradiation.altitude}
        write_results(Path(output_dir), result, config, site, irradiation, obstructed_sky, **config["output"])
    return result


def write_results(output_dir, result, config, site, irradiation, obstructed_sky, plots_html=True, plots_png=True, pdf_report=True, png_dpi=120, **kwargs):
    """Write the results into a directory: export files (results.json, config.yaml, CSV tables), and optionally the interactive plots (plots.html), PNG plots and the PDF report (report.pdf)."""
    for name, text in export_files(result, config, site, irradiation, obstructed_sky).items():
        write_text(output_dir / name, text)
    if plots_html:
        write_plots_html(output_dir / "plots.html", result_plots(result, obstructed_sky, irradiation), title=f"PV yield: tilt {result.tilt_deg:g}°, azimuth {result.azimuth_deg:g}°")
    if plots_png:
        for name, figure in static_figures(result, obstructed_sky, irradiation).items():
            figure.savefig(output_dir / f"plot_{name}.png", dpi=png_dpi)
        log.info(f"Wrote the plots as PNG files to {output_dir}")
    if pdf_report:
        (output_dir / "report.pdf").write_bytes(make_pdf_report(result, config, site, irradiation, obstructed_sky))
        log.info(f"Wrote {output_dir / 'report.pdf'}")


def write_plots_html(path, plots, title):
    """Save Bokeh plots, one below the other, as a standalone HTML file with BokehJS inlined."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save(column(list(plots.values()), sizing_mode="scale_width"), path, resources=INLINE, title=title)
    log.info(f"Wrote {path}")


def load_or_compute_obstructed_sky(sky_input, config, project=None):
    """Obstructed sky description from a JSON file (kind obstructed_sky), computed from a point cloud file, else from the project (its description, else its point cloud), or, without input, a free sky (no obstruction).

    Inputs given as files are also stored in the project.
    """
    if sky_input is None and project is not None:
        if project.has(OBSTRUCTED_SKY_NAME):
            log.info(f"Obstructed sky description from the project: {OBSTRUCTED_SKY_NAME}")
            return ObstructedSky.from_dict(json.loads(project.read_text(OBSTRUCTED_SKY_NAME)), source=f"{project.path}/{OBSTRUCTED_SKY_NAME}")
        if point_cloud_name := project.find(POINT_CLOUD_STEM):
            log.info(f"Obstructed sky from the project's point cloud: {point_cloud_name}")
            return run_obstruction(project.details.get(POINT_CLOUD_STEM, {}).get("original_filename") or point_cloud_name, config, project.read_text(point_cloud_name))
    if sky_input is None:
        log.info("No obstructed sky description given: computing without obstruction")
        return ObstructedSky.free(SkyDiscretization.from_node_count(**config["simulation"]))
    if require_file(sky_input, "point cloud or obstructed sky description").suffix.lower() == ".json":
        return ObstructedSky.from_dict(read_json(sky_input, "obstructed sky description"), source=str(sky_input))
    if project is not None:
        project.write_stem_file(POINT_CLOUD_STEM, sky_input.name, Path(sky_input))
    return run_obstruction(sky_input, config)


def weather_input(weather_path, project=None):
    """Weather file given, else the project's: returns (file name, content or None to read the file, weather source stored in the project or None); (None, None, None) if there is none."""
    if weather_path is not None:
        return weather_path, None, None
    if project is not None and (name := project.find(WEATHER_STEM)):
        details = project.details.get(WEATHER_STEM, {})
        log.info(f"Weather data from the project: {name} (originally {details.get('original_filename') or name})")
        return details.get("original_filename") or name, project.read_text(name), details.get("source")
    return None, None, None


def store_weather(project, weather_path, config):
    """Store a weather file given on the command line or downloaded in the project, with the config's weather source."""
    if project is not None and weather_path is not None:
        project.write_stem_file(WEATHER_STEM, Path(weather_path).name, Path(weather_path).read_text(encoding="utf-8"), source=config["weather"].get("source") or "auto")


def store_results(project, result, config, irradiation, obstructed_sky):
    """Write the results into the project as the browser app does: the export files under results/ and the PDF report."""
    site = {"latitude": irradiation.latitude, "longitude": irradiation.longitude, "altitude": irradiation.altitude}
    for name, text in export_files(result, config, site, irradiation, obstructed_sky).items():
        project.write(f"{RESULTS_DIRECTORY}/{name}", text)
    project.write(REPORT_NAME, make_pdf_report(result, config, site, irradiation, obstructed_sky))


def require_output(output, project, option):
    """Error unless an output file or a project is given."""
    if output is None and project is None:
        raise ValueError(f"Give {option} or a project (-p) to write the result to")


def command_obstruction(args, config, project):
    """`obstruction`: a point cloud gives the obstructed sky description; an AR recording gives the reconstructed point cloud (the obstructed sky from it follows in #49).

    Without an input file, the project's input is used: its AR recording if `sky_obstruction.method` is `ar_video` or
    the project has no point cloud, else its point cloud.
    """
    require_output(args.output, project, "-o/--output")
    use_recording = is_recording_path(args.input) if args.input is not None else (
        project is not None and project.has(RECORDING_NAME) and (config["sky_obstruction"].get("method") == "ar_video" or project.find(POINT_CLOUD_STEM) is None))
    if use_recording:
        recording, images = read_recording(args.input, project)
        points = run_reconstruction(recording, images, config)
        if args.output is not None:
            write_text(args.output, points.to_csv_text())
        if project is not None:
            project.write(RECONSTRUCTED_POINTS_NAME, points.to_csv_text())
        if args.plot is not None:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            reconstruction_figure(points).savefig(args.plot, dpi=120)
            log.info(f"Wrote {args.plot}")
        return
    if args.input is None and (project is None or project.find(POINT_CLOUD_STEM) is None):
        raise ValueError("Give a point cloud file or an AR recording, or a project (-p) with one")
    if args.input is not None:
        obstructed_sky = run_obstruction(args.input, config)
        if project is not None:
            project.write_stem_file(POINT_CLOUD_STEM, args.input.name, Path(args.input))
    else:
        name = project.find(POINT_CLOUD_STEM)
        obstructed_sky = run_obstruction(project.details.get(POINT_CLOUD_STEM, {}).get("original_filename") or name, config, project.read_text(name))
    if args.output is not None:
        write_text(args.output, to_json_text(obstructed_sky.to_dict()))
    if project is not None:
        project.write(OBSTRUCTED_SKY_NAME, to_json_text(obstructed_sky.to_dict()))


def command_irradiation(args, config, project):
    """`irradiation`: weather file (or the project's) → irradiation per sky patch, on the discretization of --sky or the project's obstructed sky description, else of the config."""
    require_output(args.output, project, "-o/--output")
    name, content, source = weather_input(args.weather, project)
    if name is None:
        raise ValueError("Give a weather file, or a project (-p) with one")
    if args.sky:
        sky = ObstructedSky.from_dict(read_json(args.sky, "obstructed sky description"), source=str(args.sky)).sky
    elif project is not None and project.has(OBSTRUCTED_SKY_NAME):
        sky = ObstructedSky.from_dict(json.loads(project.read_text(OBSTRUCTED_SKY_NAME)), source=f"{project.path}/{OBSTRUCTED_SKY_NAME}").sky
    else:
        sky = None
    irradiation = run_irradiation(name, config, sky, content, source)
    if args.output is not None:
        write_text(args.output, to_json_text(irradiation.to_dict()))
    if project is not None:
        store_weather(project, args.weather, config)
        project.write(IRRADIATION_NAME, to_json_text(irradiation.to_dict()))


def command_yield(args, config, project):
    """`yield`: irradiation per sky patch and obstructed sky description (given, else the project's; without one, no obstruction) → results."""
    if args.irradiation is not None:
        irradiation = PatchIrradiation.from_dict(read_json(args.irradiation, "irradiation per sky patch", "compute it with the `irradiation` subcommand"), source=str(args.irradiation))
    elif project is not None and project.has(IRRADIATION_NAME):
        irradiation = PatchIrradiation.from_dict(json.loads(project.read_text(IRRADIATION_NAME)), source=f"{project.path}/{IRRADIATION_NAME}")
    else:
        raise ValueError("Give the irradiation per sky patch (compute it with the `irradiation` subcommand), or a project (-p) with one")
    if args.obstructed_sky is not None:
        obstructed_sky = ObstructedSky.from_dict(read_json(args.obstructed_sky, "obstructed sky description", "compute it with the `obstruction` subcommand"), source=str(args.obstructed_sky))
    elif project is not None and project.has(OBSTRUCTED_SKY_NAME):
        obstructed_sky = ObstructedSky.from_dict(json.loads(project.read_text(OBSTRUCTED_SKY_NAME)), source=f"{project.path}/{OBSTRUCTED_SKY_NAME}")
    else:
        log.info("No obstructed sky description given: computing without obstruction")
        obstructed_sky = ObstructedSky.free(irradiation.sky)
    result = run_yield(irradiation, obstructed_sky, config, args.output_dir)
    if project is not None:
        store_results(project, result, config, irradiation, obstructed_sky)


def command_run(args, config, project):
    """`run`: all steps; inputs from the command line, else from the project; the weather is downloaded if there is none."""
    require_output(args.output_dir, project, "-d/--output-dir")
    obstructed_sky = load_or_compute_obstructed_sky(args.sky_input, config, project)
    if args.output_dir is not None:
        write_text(args.output_dir / "obstructed_sky.json", to_json_text(obstructed_sky.to_dict()))
    name, content, source = weather_input(args.weather, project)
    downloaded = None
    if name is None:
        download_dir = args.output_dir if args.output_dir is not None else Path(tempfile.mkdtemp(prefix="pv_yield_weather_"))
        name = downloaded = download_weather(config, output_dir=download_dir, reuse=args.output_dir is not None)
    irradiation = run_irradiation(name, config, obstructed_sky.sky, content, source)
    if args.output_dir is not None:
        write_text(args.output_dir / "patch_irradiation.json", to_json_text(irradiation.to_dict()))
    result = run_yield(irradiation, obstructed_sky, config, args.output_dir)
    if project is not None:
        store_weather(project, args.weather or downloaded, config)
        project.write(OBSTRUCTED_SKY_NAME, to_json_text(obstructed_sky.to_dict()))
        project.write(IRRADIATION_NAME, to_json_text(irradiation.to_dict()))
        store_results(project, result, config, irradiation, obstructed_sky)


def main(argv=None):
    """Run the command-line front end."""
    args = parse_arguments(argv)
    setup_logging(args.log_level)
    log.info(f"pv_yield_estimator {__version__}")
    project = ProjectFiles(args.project) if args.project is not None else None
    config = assemble_config(args.config, args.mod, project)
    if args.command in ("weather", "irradiation", "run", "obstruction"):
        config = resolve_site_query(config)
    log_config(config)
    if args.export_config is not None:
        write_text(args.export_config, config_to_yaml(config))
    if args.command == "obstruction":
        command_obstruction(args, config, project)
    elif args.command == "weather":
        path = download_weather(config, args.output)
        store_weather(project, path, config)
    elif args.command == "irradiation":
        command_irradiation(args, config, project)
    elif args.command == "yield":
        command_yield(args, config, project)
    elif args.command == "run":
        command_run(args, config, project)
    if project is not None and args.command is not None:
        project.write(CONFIG_NAME, config_to_yaml(config))
        project.save()


if __name__ == "__main__":
    main()
