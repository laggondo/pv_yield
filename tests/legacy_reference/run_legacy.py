"""Run the legacy code on the sample data and record its key figures in `legacy_key_figures.yaml`.

Reference only (#23): the legacy functions are called unchanged, with the parameters of the legacy main block and
the sample files from `data/`. Run from the repository root:

    python tests/legacy_reference/run_legacy.py

The recorded figures are the reference of `tests/test_legacy_comparison.py`.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPOSITORY / "legacy_code"), str(Path(__file__).parent)]
import estimate_inc_rad as legacy
import utilityLib
from meteo_data_handler import MeteoDataHandlerTMY3
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
### `utilityLib.SunPosition` calls `clean_acos`, which the sample of the legacy library lacks: arccos clipped to [-1, 1].
utilityLib.clean_acos = lambda value: np.arccos(np.clip(value, -1.0, 1.0))

### Parameters of the legacy main block.
LIDAR_YAW_DEG = -90.0 + 8.1
N_SKY_NODES = 500
MIN_POINTS_PER_PATCH = 10
ORIENTATIONS_DB = {"horizontal": (0.0, 0.0), "tilt15_south": (15.0, 0.0), "tilt30_south": (30.0, 0.0)}


def main():
    """Compute the legacy obstruction and hourly irradiance and write the key figures."""
    meteo_handler = MeteoDataHandlerTMY3(str(REPOSITORY / "data" / "Freiburg-hour.csv"))
    points_raw = legacy.load_point_cloud(str(REPOSITORY / "data" / "2026-06-01_22-25-29_red_red.csv"), max_points=None)
    rotation = Rotation.from_euler("z", -LIDAR_YAW_DEG, degrees=True).as_matrix()
    points_geo = legacy.filter_upper_hemisphere(points_raw @ rotation.T)
    point_zen, point_azm = legacy.points_to_spherical(points_geo)
    sky_disc = legacy.build_sky_discretization(n_nodes=N_SKY_NODES)
    obstruction_mask = legacy.compute_obstruction_mask(sky_disc, point_zen, point_azm, min_points_per_patch=MIN_POINTS_PER_PATCH)
    figures = {
        "parameters": {"lidar_yaw_deg": LIDAR_YAW_DEG, "n_sky_nodes": N_SKY_NODES, "min_points_per_patch": MIN_POINTS_PER_PATCH, "n_sub_steps": 6},
        "n_patches": int(len(obstruction_mask)), "n_obstructed_patches": int(obstruction_mask.sum()),
        "sky_view_factor_horizontal": float(legacy.compute_sky_view_factor(sky_disc, obstruction_mask)),
        ### Obstructed patches as sorted node triples in the new coordinate system (x east, y north), for a patch-wise comparison.
        "obstructed_patch_nodes_enu": [[[round(-float(sky_disc.nodes[node, 0]), 9), round(-float(sky_disc.nodes[node, 1]), 9), round(float(sky_disc.nodes[node, 2]), 9)] for node in simplex] for simplex in sky_disc.simplices[obstruction_mask]],
        "orientations": {},
    }
    for name, (tilt_deg, azimuth_db_deg) in ORIENTATIONS_DB.items():
        results = legacy.compute_hourly_irradiance(meteo_handler, legacy.compute_panel_normal(tilt_deg, azimuth_db_deg), obstruction_mask, sky_disc)
        annual = {key: float(np.sum(results[key]) / 1000.0) for key in ["beam_unobstructed", "beam_obstructed", "diffuse_unobstructed", "diffuse_obstructed", "total_unobstructed", "total_obstructed"]}
        annual["shading_loss"] = 1.0 - annual["total_obstructed"] / annual["total_unobstructed"]
        figures["orientations"][name] = {"tilt_deg": tilt_deg, "azimuth_compass_deg": azimuth_db_deg + 180.0, "annual_kwh_m2": annual}
        print(name, {key: round(value, 4) for key, value in annual.items()})
    print("sky view factor (horizontal):", figures["sky_view_factor_horizontal"], "obstructed patches:", figures["n_obstructed_patches"], "/", figures["n_patches"])
    (Path(__file__).parent / "legacy_key_figures.yaml").write_text(yaml.safe_dump(figures, sort_keys=False, default_flow_style=None, width=200), encoding="utf-8")


if __name__ == "__main__":
    main()
