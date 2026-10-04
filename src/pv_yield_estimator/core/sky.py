"""Sky hemisphere discretized into triangular sky patches, and the obstructed sky description.

Coordinate system: x points east, y north, z up. Azimuths follow the compass (0° = north, clockwise), zenith angles
are measured from z. The legacy code uses a different convention (Duffie-Beckman: azimuth 0° = south, x west, y south).
"""

import datetime
import logging
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import ConvexHull, KDTree

from pv_yield_estimator.file_format import FORMAT_VERSION_KEY, check_format_version

log = logging.getLogger(__name__)

OBSTRUCTED_SKY_FORMAT_VERSION = 1


def directions_from_zenith_azimuth(zenith_rad, azimuth_rad):
    """Unit vectors (x east, y north, z up) from zenith and compass azimuth angles in radians; shape (..., 3)."""
    zenith_rad, azimuth_rad = np.asarray(zenith_rad, dtype=float), np.asarray(azimuth_rad, dtype=float)
    return np.stack([np.sin(zenith_rad) * np.sin(azimuth_rad), np.sin(zenith_rad) * np.cos(azimuth_rad), np.cos(zenith_rad)], axis=-1)


def zenith_azimuth_from_directions(directions):
    """Zenith and compass azimuth angles in radians (azimuth in [0, 2π)) of direction vectors of shape (..., 3)."""
    directions = np.asarray(directions, dtype=float)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    zenith = np.arccos(np.clip(directions[..., 2], -1.0, 1.0))
    azimuth = np.mod(np.arctan2(directions[..., 0], directions[..., 1]), 2 * np.pi)
    return zenith, azimuth


def panel_normal(tilt_deg=0.0, azimuth_deg=180.0):
    """Unit normal of a panel with the given tilt (0° = horizontal) facing the given compass azimuth (180° = south)."""
    return directions_from_zenith_azimuth(np.radians(tilt_deg), np.radians(azimuth_deg))


class SkyDiscretization:
    """Upper sky hemisphere split into triangular sky patches.

    Nodes are unit vectors, triangles are triples of node indices ordered counterclockwise as seen from outside the
    sphere. Patch edges are great circle arcs, so the patches tile the hemisphere exactly (the horizon is a great
    circle, too). Everything below patch level (sun positions, LiDAR points) is mapped to patches with `find_patches`.
    """

    def __init__(self, nodes, triangles):
        self.nodes = np.asarray(nodes, dtype=float)
        self.triangles = np.array(triangles, dtype=int)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3 or self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"Sky discretization needs nodes of shape (n, 3) and triangles of shape (m, 3), got {self.nodes.shape} and {self.triangles.shape}")
        if not np.allclose(np.linalg.norm(self.nodes, axis=1), 1.0, atol=1e-9):
            raise ValueError("Sky discretization nodes must be unit vectors")
        self.nodes = self.nodes / np.linalg.norm(self.nodes, axis=1, keepdims=True)
        self._orient_triangles_outwards()
        corners = self.nodes[self.triangles]                                     ### (m, 3 corners, 3 coordinates)
        ### Barycentric coordinates of a direction d in triangle t: solve corners[t].T @ coords = d.
        self._inverse_corner_matrices = np.linalg.inv(np.transpose(corners, (0, 2, 1)))
        self._kd_tree = KDTree(self.nodes)
        triangles_per_node = [[] for _ in range(len(self.nodes))]
        for triangle_index, triangle in enumerate(self.triangles):
            for node_index in triangle:
                triangles_per_node[node_index].append(triangle_index)
        max_triangles_per_node = max(len(entry) for entry in triangles_per_node)
        ### Padded with -1; used to find candidate triangles near a direction without looping over all triangles.
        self._triangles_per_node = np.full((len(self.nodes), max_triangles_per_node), -1, dtype=int)
        for node_index, entry in enumerate(triangles_per_node):
            self._triangles_per_node[node_index, :len(entry)] = entry

    @classmethod
    def from_node_count(cls, n_sky_nodes=500, **kwargs):
        """Discretization with about `n_sky_nodes` nodes in rings of constant zenith angle, as in the legacy code.

        Rings run from the zenith down to the horizon with roughly equal node spacing, each starting at south; the
        triangles are the faces of the convex hull of the nodes, without the flat bottom faces in the horizon plane.
        For the same `n_sky_nodes`, the nodes equal those of the legacy `DensitySkyDiscretization`.
        """
        if n_sky_nodes < 5:
            raise ValueError(f"n_sky_nodes must be at least 5, got {n_sky_nodes}")
        ### Node spacing so that rings of perimeter 2π sin(zenith) with this spacing give about n_sky_nodes nodes.
        node_spacing = (np.pi + np.sqrt(np.pi ** 2 + 8 * n_sky_nodes * np.pi)) / 2.0 / n_sky_nodes
        n_rings = int(np.round(np.pi / 2 / node_spacing)) + 1
        ring_zeniths = np.linspace(0.0, np.pi / 2, n_rings)
        nodes_per_ring = np.maximum(np.round(2 * np.pi * np.sin(ring_zeniths) / node_spacing).astype(int), 1)
        nodes_per_ring[0] = 1
        zenith = np.concatenate([np.full(count, ring_zenith) for ring_zenith, count in zip(ring_zeniths, nodes_per_ring)])
        azimuth = np.concatenate([np.pi + np.arange(count) * 2 * np.pi / count for count in nodes_per_ring])
        nodes = directions_from_zenith_azimuth(zenith, azimuth)
        nodes[np.isclose(zenith, np.pi / 2), 2] = 0.0
        hull = ConvexHull(nodes)
        triangles = np.array([simplex for simplex in hull.simplices if not np.allclose(nodes[simplex, 2], 0.0)])
        sky = cls(nodes, triangles)
        log.debug(f"Sky discretization: {len(sky.nodes)} nodes, {len(sky.triangles)} patches, mean edge {np.degrees(sky.mean_edge_angle_rad()):.1f}°")
        return sky

    def _orient_triangles_outwards(self):
        """Reorder triangle corners so that each triangle is counterclockwise seen from outside the sphere."""
        a, b, c = (self.nodes[self.triangles[:, i]] for i in range(3))
        clockwise = np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c) < 0
        self.triangles[clockwise] = self.triangles[clockwise][:, [1, 0, 2]]

    @property
    def n_patches(self):
        """Number of sky patches (triangles)."""
        return len(self.triangles)

    def solid_angles(self):
        """Solid angle of each patch in steradians (Van Oosterom and Strackee); they sum to 2π over the hemisphere."""
        a, b, c = (self.nodes[self.triangles[:, i]] for i in range(3))
        numerator = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
        denominator = 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", a, c)
        return 2.0 * np.arctan2(numerator, denominator)

    def direction_integrals(self):
        """Integral of the unit direction vector over each patch, ∫ s dΩ, shape (m, 3).

        Exact for spherical triangles: half the sum over the edges of the edge angle times the unit normal of the
        edge's great circle plane (as in Lambert's formula for polygonal light sources). For an isotropic sky of
        radiance L, a surface with normal n receives L · (integral · n) from a patch that lies fully in front of it; the
        sum of the z components over the hemisphere is π.
        """
        integrals = np.zeros((self.n_patches, 3))
        for i in range(3):
            start, end = self.nodes[self.triangles[:, i]], self.nodes[self.triangles[:, (i + 1) % 3]]
            normal = np.cross(start, end)
            edge_angle = np.arctan2(np.linalg.norm(normal, axis=1), np.einsum("ij,ij->i", start, end))
            integrals += 0.5 * edge_angle[:, None] * normal / np.linalg.norm(normal, axis=1, keepdims=True)
        return integrals

    def patch_centers(self):
        """Centre direction of each patch: the normalized direction integral (mean direction over the patch)."""
        integrals = self.direction_integrals()
        return integrals / np.linalg.norm(integrals, axis=1, keepdims=True)

    def mean_edge_angle_rad(self):
        """Mean angle between the two nodes of a triangle edge, a measure of the patch size."""
        a, b = self.nodes[self.triangles], self.nodes[np.roll(self.triangles, 1, axis=1)]
        return float(np.mean(np.arccos(np.clip(np.einsum("ijk,ijk->ij", a, b), -1.0, 1.0))))

    def find_patches(self, directions, chunk_size=200_000):
        """Index of the patch containing each direction (shape (n, 3), need not be normalized); -1 below the horizon.

        Each direction maps to exactly one patch: among candidate triangles around the nearest nodes, the one with the
        largest smallest barycentric coordinate wins, so directions on a shared edge are assigned consistently. The
        rare directions not inside any candidate are checked against all patches.
        """
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        patch_indices = np.full(len(directions), -1, dtype=int)
        for start in range(0, len(directions), chunk_size):
            chunk = directions[start:start + chunk_size]
            patch_indices[start:start + chunk_size] = self._find_patches_chunk(chunk)
        return patch_indices

    def _find_patches_chunk(self, directions):
        """`find_patches` for one chunk of directions."""
        norms = np.linalg.norm(directions, axis=1)
        above_horizon = (directions[:, 2] >= 0.0) & (norms > 0.0)
        result = np.full(len(directions), -1, dtype=int)
        unit = directions[above_horizon] / norms[above_horizon, None]
        if len(unit) == 0:
            return result
        _, nearest_nodes = self._kd_tree.query(unit, k=3)
        candidates = self._triangles_per_node[nearest_nodes].reshape(len(unit), -1)
        coordinates = np.einsum("nckl,nl->nck", self._inverse_corner_matrices[np.maximum(candidates, 0)], unit)
        smallest_coordinate = np.where(candidates >= 0, coordinates.min(axis=2), -np.inf)
        best = np.argmax(smallest_coordinate, axis=1)
        found = candidates[np.arange(len(unit)), best]
        not_inside = smallest_coordinate[np.arange(len(unit)), best] < -1e-9
        if np.any(not_inside):
            log.debug(f"find_patches: {np.count_nonzero(not_inside)} directions outside the candidate patches, checking all patches")
            all_coordinates = np.einsum("tkl,nl->ntk", self._inverse_corner_matrices, unit[not_inside])
            found[not_inside] = np.argmax(all_coordinates.min(axis=2), axis=1)
        result[above_horizon] = found
        return result

    def to_dict(self):
        """Nodes (rounded to 12 decimals, enough for unit vectors) and triangles as plain lists, for JSON files."""
        return {"nodes": (np.round(self.nodes, 12) + 0.0).tolist(), "triangles": self.triangles.tolist()}

    @classmethod
    def from_dict(cls, data):
        """Inverse of `to_dict`."""
        return cls(data["nodes"], data["triangles"])

    def same_as(self, other):
        """True if both discretizations have the same nodes and triangles, so patch indices match."""
        return self.nodes.shape == other.nodes.shape and np.array_equal(self.triangles, other.triangles) and np.allclose(self.nodes, other.nodes, atol=1e-9)


class ObstructedSky:
    """Obstructed sky description: a sky discretization with one obstructed yes/no flag per patch, plus metadata.

    Independent of sun position, weather data and panel orientation. The metadata (location, date, producing method
    and its settings) is informational only; the file format is the same for all methods.
    """

    def __init__(self, sky, obstructed, metadata=None):
        self.sky = sky
        self.obstructed = np.asarray(obstructed, dtype=bool)
        if self.obstructed.shape != (sky.n_patches,):
            raise ValueError(f"Obstructed flags must have one entry per patch ({sky.n_patches}), got shape {self.obstructed.shape}")
        self.metadata = dict(metadata or {})

    @classmethod
    def free(cls, sky):
        """No obstruction at all: the default when no obstructed sky description is given."""
        return cls(sky, np.zeros(sky.n_patches, dtype=bool), {"method": "none"})

    def contributing_methods(self):
        """The methods that produced this description, in order (e.g. ["lidar", "photo"]); empty for the free sky."""
        if self.metadata.get("methods"):
            return list(self.metadata["methods"])
        return [self.metadata["method"]] if self.metadata.get("method") not in (None, "", "none") else []

    def edited(self, obstructed, methods, details=None):
        """Copy with obstructed flags marked by hand (photo method, sky map), on the same discretization (#12).

        The edits are baked into the flags; the metadata keeps the original entries and records the contributing
        methods (`methods`, joined as `method`, e.g. "lidar+photo") and one entry per editing session under `edits`
        with its date, method(s), the number of changed patches and `details` (e.g. the photos' camera views).
        """
        obstructed = np.asarray(obstructed, dtype=bool)
        if obstructed.shape != self.obstructed.shape:
            raise ValueError(f"Edited obstructed flags must have one entry per patch ({self.sky.n_patches}), got shape {obstructed.shape}")
        all_methods = self.contributing_methods() + [method for method in methods if method not in self.contributing_methods()]
        edit = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "methods": list(methods),
                "n_changed": int(np.count_nonzero(obstructed != self.obstructed)), **(details or {})}
        metadata = {**self.metadata, "method": "+".join(all_methods) or "none", "methods": all_methods, "edits": [*self.metadata.get("edits", []), edit]}
        return ObstructedSky(self.sky, obstructed, metadata)

    def obstructed_solid_angle_fraction(self):
        """Fraction of the hemisphere's solid angle that is obstructed."""
        solid_angles = self.sky.solid_angles()
        return float(solid_angles[self.obstructed].sum() / solid_angles.sum())

    def sky_view_factor(self, normal=(0.0, 0.0, 1.0)):
        """Fraction of isotropic diffuse radiation from the sky that reaches a surface with the given normal despite obstructions."""
        weights = np.maximum(self.sky.direction_integrals() @ np.asarray(normal, dtype=float), 0.0)
        return float(weights[~self.obstructed].sum() / weights.sum())

    def boundary_edges(self):
        """Edges between an obstructed and a free patch, as node index pairs of shape (k, 2): the border of the obstructed sky, e.g. for plots."""
        edges = np.sort(np.concatenate([self.sky.triangles[:, [i, (i + 1) % 3]] for i in range(3)]), axis=1)
        obstructed_count = np.tile(self.obstructed.astype(int), 3)
        unique_edges, edge_index = np.unique(edges, axis=0, return_inverse=True)
        ### Each inner edge belongs to two patches; it is on the border if exactly one of them is obstructed.
        n_patches = np.bincount(edge_index, minlength=len(unique_edges))
        n_obstructed = np.bincount(edge_index, weights=obstructed_count, minlength=len(unique_edges))
        return unique_edges[(n_patches == 2) & (n_obstructed == 1)]

    def to_dict(self):
        """Content of the JSON file, with format version."""
        return {FORMAT_VERSION_KEY: OBSTRUCTED_SKY_FORMAT_VERSION, "kind": "obstructed_sky", "metadata": self.metadata, **self.sky.to_dict(), "obstructed": self.obstructed.astype(int).tolist()}

    @classmethod
    def from_dict(cls, data, source="<dict>"):
        """Inverse of `to_dict`; checks the format version."""
        check_format_version(data, OBSTRUCTED_SKY_FORMAT_VERSION, "Obstructed sky description", source)
        if data.get("kind") != "obstructed_sky":
            raise ValueError(f"{source} is not an obstructed sky description (kind {data.get('kind')!r})")
        return cls(SkyDiscretization.from_dict(data), data["obstructed"], data.get("metadata"))


class SkyObstructionMethod(ABC):
    """Common interface of the methods producing an obstructed sky description (LiDAR, later photo, ...).

    A method is constructed with its settings; `read_input` parses its input file content, and `obstructed_sky` turns
    the input data into an `ObstructedSky` on a given discretization. `settings` is recorded in the metadata for
    information.
    """

    method_name = "abstract"

    @abstractmethod
    def read_input(self, content):
        """Parse the method's input from file content (text or file-like object), e.g. a point cloud."""

    @abstractmethod
    def obstructed_flags(self, sky, data):
        """Boolean obstructed flag per patch of `sky` from the method's input data."""

    @property
    @abstractmethod
    def settings(self):
        """The method's settings as a plain dict, recorded in the metadata."""

    def obstructed_sky(self, sky, data, metadata=None):
        """Obstructed sky description from the method's input data, with method, settings and date in the metadata."""
        obstructed = self.obstructed_flags(sky, data)
        metadata = {"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "method": self.method_name, "method_settings": self.settings, **(metadata or {})}
        result = ObstructedSky(sky, obstructed, metadata)
        log.info(f"Obstructed sky ({self.method_name}): {np.count_nonzero(obstructed)} of {sky.n_patches} patches obstructed, {result.obstructed_solid_angle_fraction():.1%} of the solid angle, horizontal sky view factor {result.sky_view_factor():.4f}")
        return result
