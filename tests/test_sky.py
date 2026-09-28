"""Tests of the sky discretization and the obstructed sky description."""

import json

import numpy as np
import pytest

from pv_yield_estimator.core.sky import ObstructedSky, SkyDiscretization, directions_from_zenith_azimuth, panel_normal, zenith_azimuth_from_directions
from pv_yield_estimator.file_format import to_json_text


@pytest.fixture(scope="module")
def sky():
    """A medium-resolution discretization."""
    return SkyDiscretization.from_node_count(300)


@pytest.mark.parametrize("n_sky_nodes", [5, 50, 500])
def test_patches_tile_the_hemisphere(n_sky_nodes):
    """Patch solid angles sum to 2π and the z components of the direction integrals to π (∫ cos θ dΩ)."""
    sky = SkyDiscretization.from_node_count(n_sky_nodes)
    assert sky.solid_angles().sum() == pytest.approx(2 * np.pi, rel=1e-12)
    assert sky.direction_integrals().sum(axis=0) == pytest.approx([0.0, 0.0, np.pi], abs=1e-12)
    assert np.all(sky.patch_centers()[:, 2] > 0)


def test_direction_integral_of_small_patch_is_center_times_solid_angle(sky):
    """For small patches the direction integral is about the solid angle times the centre direction."""
    integrals = sky.direction_integrals()
    np.testing.assert_allclose(np.linalg.norm(integrals, axis=1), sky.solid_angles(), rtol=1e-2)


def test_every_direction_maps_to_exactly_one_patch(sky):
    """Random directions, nodes and edge midpoints map to a valid patch that contains them; below the horizon gives -1."""
    rng = np.random.default_rng(1)
    random_directions = rng.normal(size=(20000, 3))
    random_directions[:, 2] = np.abs(random_directions[:, 2])
    edge_midpoints = (sky.nodes[sky.triangles] + sky.nodes[np.roll(sky.triangles, 1, axis=1)]).reshape(-1, 3)
    directions = np.concatenate([random_directions, sky.nodes, edge_midpoints, [[1.0, 0.0, 0.0], [0.0, 0.0, 5.0]]])
    patches = sky.find_patches(directions)
    assert np.all((patches >= 0) & (patches < sky.n_patches))
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    barycentric = np.einsum("nkl,nl->nk", np.linalg.inv(np.transpose(sky.nodes[sky.triangles[patches]], (0, 2, 1))), unit)
    assert barycentric.min() > -1e-9
    assert np.all(sky.find_patches([[0.3, 0.2, -0.1], [0.0, 0.0, -1.0]]) == -1)


def test_random_directions_spread_by_solid_angle(sky):
    """Uniformly distributed directions fall into the patches in proportion to their solid angles."""
    rng = np.random.default_rng(2)
    directions = rng.normal(size=(400000, 3))
    patches = sky.find_patches(directions)
    fractions = np.bincount(patches[patches >= 0], minlength=sky.n_patches) / np.count_nonzero(patches >= 0)
    np.testing.assert_allclose(fractions, sky.solid_angles() / (2 * np.pi), atol=5 * np.sqrt(fractions.max() / 200000))


def test_angles_and_panel_normal_conventions():
    """Compass azimuth: east is +x, north +y; a south-facing tilted panel's normal points south and up."""
    np.testing.assert_allclose(directions_from_zenith_azimuth(np.pi / 2, np.pi / 2), [1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(panel_normal(90.0, 0.0), [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(panel_normal(30.0, 180.0), [0.0, -0.5, np.sqrt(3) / 2], atol=1e-15)
    zenith, azimuth = zenith_azimuth_from_directions([[-1.0, 0.0, 1.0]])
    assert np.degrees(zenith[0]) == pytest.approx(45.0) and np.degrees(azimuth[0]) == pytest.approx(270.0)


def test_obstructed_sky_json_round_trip_and_sky_view_factor(sky):
    """The obstructed sky description survives a JSON round trip; the sky view factor follows the obstructed half."""
    south_half = sky.patch_centers()[:, 1] < 0
    obstructed_sky = ObstructedSky(sky, south_half, {"method": "test"})
    loaded = ObstructedSky.from_dict(json.loads(to_json_text(obstructed_sky.to_dict())), source="test.json")
    assert loaded.sky.same_as(sky)
    np.testing.assert_array_equal(loaded.obstructed, south_half)
    assert loaded.metadata == {"method": "test"}
    assert obstructed_sky.sky_view_factor() == pytest.approx(0.5, abs=0.02)
    assert obstructed_sky.sky_view_factor(panel_normal(90.0, 180.0)) < 0.05
    assert obstructed_sky.sky_view_factor(panel_normal(90.0, 0.0)) > 0.95
    assert ObstructedSky(sky, np.zeros(sky.n_patches, bool)).sky_view_factor(panel_normal(40.0, 123.0)) == 1.0


def test_obstructed_sky_rejects_wrong_files(sky):
    """Wrong kind, missing version and wrong flag count raise errors naming the problem."""
    data = ObstructedSky(sky, np.zeros(sky.n_patches, bool)).to_dict()
    with pytest.raises(ValueError, match="not an obstructed sky"):
        ObstructedSky.from_dict(data | {"kind": "patch_irradiation"}, source="x.json")
    with pytest.raises(ValueError, match="format_version"):
        ObstructedSky.from_dict({key: value for key, value in data.items() if key != "format_version"})
    with pytest.raises(ValueError, match="one entry per patch"):
        ObstructedSky(sky, [True, False])
