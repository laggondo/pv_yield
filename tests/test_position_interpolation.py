"""Tests of the obstructed sky at other panel positions, interpolated from the sky maps of photo groups (#54), on synthetic scenes."""

from pathlib import Path

import numpy as np
import pytest

from pv_yield_estimator.core.irradiation import compute_patch_irradiation
from pv_yield_estimator.core.panel_yield import compute_panel_radiation
from pv_yield_estimator.core.position_interpolation import PhotoGroup, PositionInterpolation, filled_patches, moved_directions, obstruction_regions
from pv_yield_estimator.core.sky import ObstructedSky, SkyDiscretization
from pv_yield_estimator.core.weather import load_weather, resolve_site
from synthetic_scene import SyntheticScene, balcony_scene, courtyard_scene, garage_scene

PVGIS_PATH = Path(__file__).resolve().parents[1] / "data" / "Freiburg-pvgis-tmy.csv"
### Scenes with the position of group B (A at the origin) and the panel positions to check: between the groups, in front of the line between them, and beyond B.
SCENES = {
    "garage": (garage_scene(), (6.0, 0.0, 0.0), [(1.5, 0.0, 0.0), (3.0, 0.0, 0.0), (4.5, 0.0, 0.0), (3.0, -1.0, 0.0), (7.0, 0.0, 0.0)]),
    "courtyard": (courtyard_scene(), (3.0, 0.0, 0.0), [(1.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.0, 0.0, 0.0), (1.5, -1.0, 0.0), (4.0, 0.0, 0.0)]),
    "balcony": (balcony_scene(), (4.0, 0.0, 0.0), [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, -0.5, 0.0), (5.0, 0.0, 0.0)]),
}
PANEL = {"tilt_deg": 30.0, "azimuth_deg": 180.0}


@pytest.fixture(scope="module")
def sky():
    """The default resolution."""
    return SkyDiscretization.from_node_count(500)


@pytest.fixture(scope="module")
def irradiation(sky):
    """Irradiation per sky patch for the sample weather data (Freiburg)."""
    weather = load_weather(PVGIS_PATH.read_text(encoding="utf-8"), source="pvgis_tmy")
    return compute_patch_irradiation(weather, sky, **resolve_site(weather))


def groups_for(scene, sky, position_b, position_error_b=(0.0, 0.0, 0.0)):
    """Photo groups A (origin) and B with the scene's exact sky maps; B's position as placed on the map may be off by `position_error_b`."""
    return [PhotoGroup("A", scene.obstructed_sky(sky, (0.0, 0.0, 0.0), "A")), PhotoGroup("B", scene.obstructed_sky(sky, position_b, "B"), np.add(position_b, position_error_b))]


def comparison_rows(sky, irradiation, position_error_b=(0.0, 0.0, 0.0)):
    """Interpolated and nearest-group obstruction against the exact one at each scene's panel positions: share of wrong patches (by solid angle), horizontal sky view factor and annual radiation on a south-facing panel (tilt 30°)."""
    solid_angles = sky.solid_angles()
    annual = lambda obstructed_sky: compute_panel_radiation(irradiation, obstructed_sky, **PANEL)["total_obstructed"].sum() / 1000.0
    rows = []
    for name, (scene, position_b, panel_positions) in SCENES.items():
        groups = groups_for(scene, sky, position_b, position_error_b)
        interpolation = PositionInterpolation(groups)
        for panel_position in panel_positions:
            exact = scene.obstructed_sky(sky, panel_position)
            interpolated = interpolation.obstructed_sky(panel_position)
            nearest = groups[interpolation.nearest_group_index(np.array(panel_position))].obstructed_sky
            wrong = lambda candidate: float((candidate.obstructed != exact.obstructed) @ solid_angles / solid_angles.sum())
            rows.append({"scene": name, "panel": panel_position, "wrong_interpolated": wrong(interpolated), "wrong_nearest": wrong(nearest),
                         "svf_exact": exact.sky_view_factor(), "svf_interpolated": interpolated.sky_view_factor(), "svf_nearest": nearest.sky_view_factor(),
                         "annual_exact": annual(exact), "annual_interpolated": annual(interpolated), "annual_nearest": annual(nearest)})
    return rows


def print_rows(title, rows):
    """Print the comparison as a table (shown with pytest -s), for the pull request."""
    print(f"\n{title}\n| scene | panel position (m) | wrong patches interpolated / nearest | sky view factor exact / interpolated / nearest | annual radiation (kWh/m²) exact / interpolated / nearest |\n|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['scene']} | {row['panel'][0]:g}, {row['panel'][1]:g} | {row['wrong_interpolated']:.1%} / {row['wrong_nearest']:.1%} | {row['svf_exact']:.3f} / {row['svf_interpolated']:.3f} / {row['svf_nearest']:.3f} | "
              f"{row['annual_exact']:.0f} / {row['annual_interpolated']:.0f} / {row['annual_nearest']:.0f} |")


def test_moved_directions_follow_the_geometry():
    """A point at distance r seen from the origin is seen from another position in the direction of the moved point; q = 0 doesn't move."""
    direction = np.array([0.6, 0.0, 0.8])
    baseline = np.array([2.0, -1.0, 0.5])
    expected = 5.0 * direction - baseline
    assert moved_directions(direction, 1.0 / 5.0, baseline) == pytest.approx(expected / np.linalg.norm(expected))
    assert moved_directions(direction, 0.0, baseline) == pytest.approx(direction)


def test_filled_patches_and_regions(sky):
    """Unmoved patches fill exactly themselves, also when turned over; regions are the connected groups of obstructed patches."""
    flags = np.zeros(sky.n_patches, dtype=bool)
    flags[[0, 1, 2, 500, 900]] = True
    corners = sky.nodes[sky.triangles[flags]]
    assert np.array_equal(filled_patches(sky, corners), flags)
    assert np.array_equal(filled_patches(sky, corners[:, ::-1]), flags)
    scene = SyntheticScene([((-2.0, -9.0, -1.0), (2.0, -6.0, 4.0)), ((6.0, 3.0, -1.0), (8.0, 5.0, 3.0))])
    regions = obstruction_regions(scene.obstructed_sky(sky, (0.0, 0.0, 0.0)))
    assert len(regions) == 2 and sum(len(region) for region in regions) == np.count_nonzero(scene.obstructed_flags(sky, (0.0, 0.0, 0.0)))


def test_group_positions_give_their_own_maps(sky):
    """At a group's position, the interpolated obstruction is that group's sky map, with the groups and regions in the metadata."""
    scene, position_b, _ = SCENES["garage"]
    groups = groups_for(scene, sky, position_b)
    interpolation = PositionInterpolation(groups)
    at_a, at_b = interpolation.obstructed_sky((0.0, 0.0, 0.0)), interpolation.obstructed_sky(position_b)
    assert np.array_equal(at_a.obstructed, groups[0].obstructed_sky.obstructed) and np.array_equal(at_b.obstructed, groups[1].obstructed_sky.obstructed)
    metadata = at_b.metadata["position_interpolation"]
    assert at_b.metadata["method"] == "photo+position_interpolation" and metadata["source_group"] == "B" and [group["name"] for group in metadata["groups"]] == ["A", "B"]
    assert metadata["regions"] and all({"distance_m", "azimuth_deg", "elevation_deg", "matched"} <= set(region) for region in metadata["regions"])
    ### A single group: its map, wherever the panel is.
    single = PositionInterpolation(groups[:1]).obstructed_sky((3.0, 0.0, 0.0))
    assert np.array_equal(single.obstructed, groups[0].obstructed_sky.obstructed)


def test_distance_of_a_single_object(sky):
    """A box 6–9 m to the south, seen from two positions 4 m apart: its distance (per azimuth sector) is found, and the obstruction between the positions has far fewer wrong patches than the nearest group's."""
    scene = SyntheticScene([((-2.0, -9.0, -1.0), (2.0, -6.0, 4.0))])
    interpolation = PositionInterpolation(groups_for(scene, sky, (4.0, 0.0, 0.0)))
    sectors = interpolation.region_summary(0)
    assert {sector["region"] for sector in sectors} == {0} and all(sector["matched"] and 5.0 < sector["distance_m"] < 10.0 for sector in sectors)
    for position in [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]:
        exact = scene.obstructed_flags(sky, position)
        nearest = interpolation.groups[interpolation.nearest_group_index(np.array(position))].obstructed_sky.obstructed
        assert np.count_nonzero(interpolation.obstructed_sky(position).obstructed != exact) < 0.4 * np.count_nonzero(nearest != exact)


def test_far_objects_stay_and_unmatched_regions_stay_in_place(sky):
    """A far horizon profile gets no shift; an object seen from group A only stays in place."""
    far = SyntheticScene(horizon_profile=lambda azimuth_deg: 8.0 + 0.0 * azimuth_deg)
    groups = groups_for(far, sky, (3.0, 0.0, 0.0))
    assert np.array_equal(PositionInterpolation(groups).obstructed_sky((1.5, 0.0, 0.0)).obstructed, groups[0].obstructed_sky.obstructed)
    near = SyntheticScene([((-1.0, -6.0, 0.0), (1.0, -5.0, 2.0))])
    only_a = [PhotoGroup("A", near.obstructed_sky(sky, (0.0, 0.0, 0.0))), PhotoGroup("B", ObstructedSky(sky, np.zeros(sky.n_patches, dtype=bool), {"method": "photo"}), (3.0, 0.0, 0.0))]
    interpolation = PositionInterpolation(only_a)
    assert not interpolation.region_summary(0)[0]["matched"]
    assert np.array_equal(interpolation.obstructed_sky((1.0, 0.0, 0.0)).obstructed, only_a[0].obstructed_sky.obstructed)


def test_three_groups_and_errors(sky):
    """Three groups: the nearest is the source, distances from both other groups; different discretizations and missing descriptions raise helpful errors."""
    scene = garage_scene()
    groups = groups_for(scene, sky, (6.0, 0.0, 0.0)) + [PhotoGroup("C", scene.obstructed_sky(sky, (3.0, -2.0, 0.0)), (3.0, -2.0, 0.0))]
    interpolation = PositionInterpolation(groups)
    result = interpolation.obstructed_sky((3.0, -1.5, 0.0))
    assert result.metadata["position_interpolation"]["source_group"] == "C"
    assert set(interpolation.region_summary(2)[0]["scores"]) == {"A", "B"}
    exact = scene.obstructed_flags(sky, (3.0, -1.5, 0.0))
    nearest = groups[2].obstructed_sky.obstructed
    assert np.count_nonzero(result.obstructed != exact) <= np.count_nonzero(nearest != exact) + 5
    with pytest.raises(ValueError, match="different sky discretizations"):
        PositionInterpolation([groups[0], PhotoGroup("D", ObstructedSky.free(SkyDiscretization.from_node_count(100)))])
    with pytest.raises(ValueError, match="at least one photo group"):
        PositionInterpolation([PhotoGroup("A", None)])
    with pytest.raises(ValueError, match="unique"):
        PositionInterpolation([groups[0], groups[0]])


def test_interpolation_against_exact_maps(sky, irradiation):
    """On the synthetic scenes, the interpolated obstruction is closer to the exact one than the nearest group's map, summed over the positions between the groups."""
    rows = comparison_rows(sky, irradiation)
    print_rows("Interpolated vs. nearest group (exact positions)", rows)
    between = [row for row in rows if row["panel"][0] > 0 and row["panel"][0] < SCENES[row["scene"]][1][0]]
    error = lambda kind, key: sum(abs(row[f"{key}_{kind}"] - row[f"{key}_exact"]) for row in between)
    assert error("interpolated", "svf") < 0.7 * error("nearest", "svf")
    assert error("interpolated", "annual") < 0.7 * error("nearest", "annual")
    assert sum(row["wrong_interpolated"] for row in between) < sum(row["wrong_nearest"] for row in between)
    garage = [row for row in between if row["scene"] == "garage"]
    assert all(abs(row["annual_interpolated"] - row["annual_exact"]) <= abs(row["annual_nearest"] - row["annual_exact"]) + 5.0 for row in garage)


def test_interpolation_with_position_error(sky, irradiation):
    """Group B placed 0.5 m off on the map (along and across the line between the groups): the interpolation stays better than the nearest group."""
    for error_b in [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0)]:
        rows = comparison_rows(sky, irradiation, error_b)
        print_rows(f"Interpolated vs. nearest group, B placed {error_b[:2]} m off", rows)
        between = [row for row in rows if row["panel"][0] > 0 and row["panel"][0] < SCENES[row["scene"]][1][0]]
        error = lambda kind: sum(abs(row[f"annual_{kind}"] - row["annual_exact"]) for row in between)
        assert error("interpolated") < error("nearest")
