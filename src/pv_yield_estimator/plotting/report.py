"""PDF report: key figures, monthly values, orientation comparison and config on the first pages, then the static plots.

Built with matplotlib's PDF backend only (no pyplot, no LaTeX), so it works in the CLI and in the browser (Pyodide).
Returns the PDF as bytes; the front ends write or download it.
"""

import datetime
import io

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from pv_yield_estimator import __version__
from pv_yield_estimator.config import config_to_yaml
from pv_yield_estimator.core.orientation import SHADING_CASES
from pv_yield_estimator.plotting.interactive import MONTH_NAMES
from pv_yield_estimator.plotting.static import PAGE_SIZE, static_figures

### Annual key figures in the report table: label (with unit), key prefix, key suffix, decimals; one column per case.
KEY_FIGURE_ROWS = (("Radiation on the panel, total (kWh/m²)", "annual_total", "_kwh_m2", 1), ("Radiation on the panel, direct (kWh/m²)", "annual_direct", "_kwh_m2", 1),
                   ("Radiation on the panel, diffuse (kWh/m²)", "annual_diffuse", "_kwh_m2", 1), ("PV yield (kWh)", "annual_yield", "_kwh", 1), ("Specific yield (kWh/kWp)", "specific_yield", "_kwh_kwp", 0))
SINGLE_KEY_FIGURES = (("Shading loss, total (%)", "shading_loss", 100.0, 1), ("Shading loss, direct (%)", "direct_shading_loss", 100.0, 1), ("Sky view factor of the panel", "sky_view_factor", 1.0, 3),
                      ("Sky view factor, horizontal", "sky_view_factor_horizontal", 1.0, 3), ("Rated power (kWp)", "rated_power_kwp", 1.0, 3))


def table_axes(figure, rect, title):
    """Axes without frame for a table or text block at `rect` (left, bottom, width, height in figure fractions)."""
    axes = figure.add_axes(rect)
    axes.axis("off")
    axes.set_title(title, loc="left", fontsize=11, fontweight="bold")
    return axes


def draw_table(axes, header, rows, column_widths=None, font_size=8):
    """A plain table filling the axes from the top, header row bold, numbers right-aligned."""
    table = axes.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="right", colWidths=column_widths, edges="horizontal")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.25)
    for (row, column), cell in table.get_celld().items():
        if column == 0:
            cell.set_text_props(ha="left")
        if row == 0:
            cell.set_text_props(fontweight="bold")
    return table


def summary_page(result, config, site, irradiation=None, obstructed_sky=None):
    """First page: inputs, key figures and average daily values per month."""
    figure = Figure(figsize=PAGE_SIZE)
    figure.text(0.04, 0.95, "PV yield estimate", fontsize=18, fontweight="bold")
    weather = irradiation.metadata if irradiation is not None else {}
    obstruction = obstructed_sky.metadata if obstructed_sky is not None else {}
    optimized = f" (optimized: {', '.join(name.removesuffix('_deg') for name in result.optimized_angles)})" if result.optimized_angles else ""
    lines = [f"Site: {site['latitude']:.4f}° N, {site['longitude']:.4f}° E, {site['altitude']:.0f} m",
             f"Weather: {weather.get('weather_name', '?')} ({weather.get('weather_source', '?')}{', file ' + weather['input_file'] if weather.get('input_file') else ''})",
             f"Obstructed sky: method {obstruction.get('method', '?')}{', from ' + obstruction['input_file'] if obstruction.get('input_file') else ''}{', created ' + obstruction['created'] if obstruction.get('created') else ''}",
             f"Panel: tilt {result.tilt_deg:g}°, azimuth {result.azimuth_deg:g}°{optimized}; area {result.area_m2:g} m², efficiency {result.efficiency:g}, performance ratio {result.performance_ratio:g}",
             f"Created {datetime.datetime.now().astimezone():%Y-%m-%d %H:%M} with pv_yield_estimator {__version__}"]
    figure.text(0.04, 0.915, "\n".join(lines), fontsize=9, va="top", linespacing=1.5)
    figures = result.key_figures()
    rows = [[label, *(f"{figures[f'{prefix}_{case}{suffix}']:.{digits}f}" for case in SHADING_CASES)] for label, prefix, suffix, digits in KEY_FIGURE_ROWS]
    rows += [[label, "", f"{figures[key] * factor:.{digits}f}"] for label, key, factor, digits in SINGLE_KEY_FIGURES]
    draw_table(table_axes(figure, (0.04, 0.05, 0.42, 0.62), "Annual key figures"), ["", *SHADING_CASES], rows, column_widths=[0.62, 0.19, 0.19])
    monthly = result.monthly_daily_average()
    rows = [[MONTH_NAMES[month - 1], *(f"{row[f'total_{case}']:.2f}" for case in SHADING_CASES), *(f"{row[f'yield_{case}']:.3f}" for case in SHADING_CASES)] for month, row in monthly.iterrows()]
    header = ["Month", "radiation\nunobstructed\n(kWh/m²/d)", "radiation\nobstructed\n(kWh/m²/d)", "yield\nunobstructed\n(kWh/d)", "yield\nobstructed\n(kWh/d)"]
    table = draw_table(table_axes(figure, (0.52, 0.05, 0.44, 0.62), "Average daily values per month"), header, rows, column_widths=[0.12, 0.22, 0.22, 0.22, 0.22])
    for column in range(5):
        table[0, column].set_height(table[0, column].get_height() * 2.6)
    return figure


def orientation_and_config_page(result, config):
    """Second page: orientation comparison table and the config used."""
    figure = Figure(figsize=PAGE_SIZE)
    header = ["", "tilt (°)", "azimuth (°)", "radiation\nunobstructed\n(kWh/m²)", "radiation\nobstructed\n(kWh/m²)", "yield\nobstructed\n(kWh)", "specific yield\nobstructed\n(kWh/kWp)", "shading\nloss (%)"]
    rows = [[row["label"], f"{row['tilt_deg']:g}", f"{row['azimuth_deg']:g}", f"{row['annual_total_unobstructed_kwh_m2']:.1f}", f"{row['annual_total_obstructed_kwh_m2']:.1f}",
             f"{row['annual_yield_obstructed_kwh']:.1f}", f"{row['specific_yield_obstructed_kwh_kwp']:.0f}", f"{100 * row['shading_loss']:.1f}"] for row in result.orientation_comparison]
    table = draw_table(table_axes(figure, (0.04, 0.55, 0.92, 0.38), "Orientation comparison (azimuth: compass, 180 = south)"), header, rows, column_widths=[0.2, 0.08, 0.1, 0.13, 0.13, 0.12, 0.14, 0.1])
    for column in range(len(header)):
        table[0, column].set_height(table[0, column].get_height() * 2.6)
    axes = table_axes(figure, (0.04, 0.03, 0.92, 0.45), "Config used")
    config_lines = config_to_yaml(config).splitlines()
    ### Two columns, split at the top-level section closest to the middle.
    section_starts = [index for index, line in enumerate(config_lines) if line and not line.startswith(" ") and not line.startswith("- ")]
    half = min(section_starts, key=lambda index: abs(index - len(config_lines) / 2))
    for column, lines in enumerate((config_lines[:half], config_lines[half:])):
        axes.text(0.5 * column, 1.0, "\n".join(lines), family="monospace", fontsize=7, va="top", transform=axes.transAxes)
    return figure


def pdf_report(result, config, site, irradiation, obstructed_sky):
    """The PDF report as bytes: summary page, orientation comparison and config, then the static plots one per page."""
    buffer = io.BytesIO()
    with PdfPages(buffer, metadata={"Title": "PV yield estimate", "Creator": f"pv_yield_estimator {__version__}"}) as pdf:
        pdf.savefig(summary_page(result, config, site, irradiation, obstructed_sky))
        pdf.savefig(orientation_and_config_page(result, config))
        for figure in static_figures(result, obstructed_sky, irradiation).values():
            pdf.savefig(figure)
    return buffer.getvalue()
