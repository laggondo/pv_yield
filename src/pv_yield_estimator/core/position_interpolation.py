"""Obstructed sky at other panel positions, from the sky maps of a few photo groups (#54).

Each photo group gives a complete obstructed sky description (sky map) seen from one camera position; the positions
are offsets east/north/up in metres relative to group A. Moving from one position to another, every obstruction point
seen in direction d at distance r shifts along the great circle through d and the baseline direction b: from the new
position it is seen in direction d - q b (normalized), with q = 1/r the inverse distance (q = 0: infinitely far, no
shift). Only q is unknown. So the obstruction at another position follows from one distance per obstruction region:

1. Regions: connected groups of obstructed patches (sharing an edge) in each group's sky map.
2. Distance per region: candidate inverse distances (distances log-spaced between `min_distance_m` and
   `max_distance_m`, plus infinity) are tried by moving the region from its group's position to another group's and
   scoring the overlap with that group's obstructed patches (intersection over union, by solid angle, within the band
   the region can reach at all). Within `score_tolerance` of the best, the farthest distance wins (smallest shift).
   Regions below `min_score` have no match.
3. Sectors: matched regions spanning several azimuth sectors (typically one connected horizon profile with near and
   far objects on it) are split into sectors of `sector_width_deg`, each with its own distance. Starting from the
   region's distance, the sectors are refined together (coordinate descent): each takes the distance for which the
   whole moved map agrees best with the other group's map around the sector, so neighbouring sectors count, too.
   With more than two groups, the median over the other groups is taken.
4. Moving to the panel position P: the region's patch nodes (not only the centres) are moved with its distance, and
   the moved triangles are filled into patches (patch centres inside a moved triangle, plus the patch of each moved
   triangle's centre). This avoids holes when a region grows as P gets closer to it. The source is the group nearest P.
5. The obstruction at P is the union of the moved regions; regions without a good match (seen from one position only)
   stay in place (q = 0), which errs on the side of more shade.

Robustness over accuracy: distances are only as good as the patch outlines and the positions. Interpolating between
groups keeps errors within the measured shift; extrapolating beyond them makes errors grow.
"""

import logging

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pv_yield_estimator.core.sky import ObstructedSky, zenith_azimuth_from_directions

log = logging.getLogger(__name__)

### Directions closer than this (as vector length before normalizing) to the moving direction are left unmoved.
DEGENERATE_LENGTH = 1e-9


class PhotoGroup:
    """One photo group: its name (A, B, ...), its obstructed sky description and its position (east, north, up in metres relative to group A)."""

    def __init__(self, name, obstructed_sky, offset_m=(0.0, 0.0, 0.0)):
        self.name = str(name)
        self.obstructed_sky = obstructed_sky
        self.offset_m = np.zeros(3) if offset_m is None else np.asarray(list(offset_m) + [0.0] * (3 - len(offset_m)), dtype=float)
        if self.offset_m.shape != (3,):
            raise ValueError(f"Photo group {self.name}: offset_m must be [east, north] or [east, north, up] in metres, got {offset_m!r}")


def moved_directions(directions, inverse_distance, baseline):
    """Directions (unit vectors, shape (..., 3)) of obstruction points at distance 1/inverse_distance, seen after moving by `baseline` (metres)."""
    moved = np.asarray(directions, dtype=float) - inverse_distance * np.asarray(baseline, dtype=float)
    length = np.linalg.norm(moved, axis=-1, keepdims=True)
    return np.where(length > DEGENERATE_LENGTH, moved / np.maximum(length, DEGENERATE_LENGTH), directions)


def filled_patches(sky, corners, centers=None):
    """Flag per patch of `sky` covered by spherical triangles given by their corner directions (shape (k, 3, 3)).

    A patch counts as covered if its centre lies inside one of the triangles, or if it contains a triangle's centre
    (so that small triangles never get lost). The triangles may be turned over by the move (seen from behind); their
    orientation does not matter.
    """
    covered = np.zeros(sky.n_patches, dtype=bool)
    if len(corners) == 0:
        return covered
    centers = sky.patch_centers() if centers is None else centers
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    orientation = np.sign(np.einsum("ij,ij->i", a, np.cross(b, c)))
    orientation[orientation == 0] = 1.0
    inside = np.ones((len(corners), sky.n_patches), dtype=bool)
    for start, end in ((a, b), (b, c), (c, a)):
        inside &= (np.cross(start, end) * orientation[:, None]) @ centers.T >= -1e-12
    covered |= inside.any(axis=0)
    triangle_centers = a + b + c
    patches = sky.find_patches(triangle_centers)
    covered[patches[patches >= 0]] = True
    return covered


def obstruction_regions(obstructed_sky):
    """Connected regions of obstructed patches (patches sharing an edge), as a list of arrays of patch indices, largest first."""
    sky, obstructed = obstructed_sky.sky, obstructed_sky.obstructed
    first, second = patch_neighbours(sky).T
    both_obstructed = obstructed[first] & obstructed[second]
    adjacency = coo_matrix((np.ones(np.count_nonzero(both_obstructed)), (first[both_obstructed], second[both_obstructed])), shape=(sky.n_patches, sky.n_patches))
    _, labels = connected_components(adjacency, directed=False)
    regions = [np.flatnonzero((labels == label) & obstructed) for label in np.unique(labels[obstructed])]
    return sorted(regions, key=len, reverse=True)


def patch_neighbours(sky):
    """Pairs of patches sharing an edge, shape (k, 2)."""
    edges = np.concatenate([np.sort(sky.triangles[:, [i, (i + 1) % 3]], axis=1) for i in range(3)])
    patch_of_edge = np.tile(np.arange(sky.n_patches), 3)
    _, edge_index = np.unique(edges, axis=0, return_inverse=True)
    edge_index = edge_index.ravel()
    order = np.argsort(edge_index, kind="stable")
    sorted_edges, sorted_patches = edge_index[order], patch_of_edge[order]
    shared = sorted_edges[1:] == sorted_edges[:-1]
    return np.stack([sorted_patches[:-1][shared], sorted_patches[1:][shared]], axis=1)


def sector_index(directions, sector_width_deg=15.0):
    """Index of the azimuth sector (about `sector_width_deg` wide) of each direction."""
    _, azimuth = zenith_azimuth_from_directions(directions)
    n_sectors = max(int(np.round(360.0 / sector_width_deg)), 1)
    return np.minimum((np.degrees(azimuth) / 360.0 * n_sectors).astype(int), n_sectors - 1)


class PositionInterpolation:
    """Obstructed sky for any panel position from the sky maps of two or more photo groups (#54).

    The distances are found once per source group (lazily) and reused for every panel position, so moving the panel
    is fast. Settings (from the config's `sky_obstruction.interpolation`): candidate distances log-spaced from
    `min_distance_m` to `max_distance_m` (`n_distances` values, plus infinity); `min_score`, the intersection over
    union a region needs for a match; `score_tolerance`, within which the farther distance is preferred;
    `sector_width_deg` for splitting regions into azimuth sectors (0: no sectors), refined in `n_sweeps` sweeps.
    """

    def __init__(self, groups, min_distance_m=1.0, max_distance_m=200.0, n_distances=40, min_score=0.3, score_tolerance=0.005, sector_width_deg=30.0, n_sweeps=3, **kwargs):
        self.groups = [group for group in groups if group.obstructed_sky is not None]
        if not self.groups:
            raise ValueError("Interpolating the obstruction needs at least one photo group with an obstructed sky description")
        names = [group.name for group in self.groups]
        if len(set(names)) != len(names):
            raise ValueError(f"Photo group names must be unique, got {names}")
        sky = self.groups[0].obstructed_sky.sky
        for group in self.groups[1:]:
            if not group.obstructed_sky.sky.same_as(sky):
                raise ValueError(f"Photo groups {self.groups[0].name} and {group.name} use different sky discretizations ({sky.n_patches} and {group.obstructed_sky.sky.n_patches} patches); "
                                 "mark all groups on the same resolution (sky nodes)")
        self.sky = sky
        self.inverse_distances = np.concatenate([[0.0], 1.0 / np.geomspace(max_distance_m, min_distance_m, n_distances)])
        self.min_score, self.score_tolerance, self.sector_width_deg, self.n_sweeps = min_score, score_tolerance, sector_width_deg, n_sweeps
        self.settings = {"min_distance_m": min_distance_m, "max_distance_m": max_distance_m, "n_distances": n_distances, "min_score": min_score, "score_tolerance": score_tolerance,
                         "sector_width_deg": sector_width_deg, "n_sweeps": n_sweeps}
        self._centers = sky.patch_centers()
        self._solid_angles = sky.solid_angles()
        self._neighbours = patch_neighbours(sky)
        self._units = {}

    def moved_region(self, patches, inverse_distance, baseline):
        """Flag per patch covered by the region (patch indices) of a sky map after moving by `baseline` (metres) with the given inverse distance."""
        corners = moved_directions(self.sky.nodes[self.sky.triangles[patches]], inverse_distance, baseline)
        return filled_patches(self.sky, corners, self._centers)

    def dilated(self, mask):
        """The patches flagged in `mask` plus their neighbours (sharing an edge)."""
        result = mask.copy()
        first, second = self._neighbours.T
        result[second[mask[first]]] = True
        result[first[mask[second]]] = True
        return result

    def region_scores(self, patches, baseline, target_obstructed):
        """Intersection over union (by solid angle) of the region moved with each candidate inverse distance and the target's obstructed patches within the band the region can reach at all."""
        moved = np.array([self.moved_region(patches, inverse_distance, baseline) for inverse_distance in self.inverse_distances])
        reachable_target = target_obstructed & moved.any(axis=0)
        intersection = (moved & target_obstructed) @ self._solid_angles
        union = (moved | reachable_target) @ self._solid_angles
        return np.where(union > 0, intersection / np.where(union > 0, union, 1.0), 0.0)

    def farthest_best(self, scores):
        """Index of the farthest candidate (smallest inverse distance) scoring within the tolerance of the best."""
        return int(np.flatnonzero(scores >= scores.max() - self.score_tolerance)[0])

    def refine_tiles(self, tiles, chosen, refine, baseline, target_obstructed):
        """Candidate index per tile (region or sector), refined for the tiles flagged in `refine` by coordinate descent, starting from `chosen`.

        In each sweep, each tile in turn takes the candidate for which the whole moved map (all tiles with their
        current candidates) agrees best with the target around the tile (the patches it can reach, and their
        neighbours), so neighbouring sectors are accounted for; the farthest within the tolerance wins.

        Returns the candidate indices and whether the agreement depended on the candidate (determined).
        """
        candidates = range(len(self.inverse_distances))
        moved = [np.array([self.moved_region(patches, self.inverse_distances[candidate], baseline) for candidate in candidates]) if refine[index] else None for index, (_, patches) in enumerate(tiles)]
        fixed = [self.moved_region(patches, self.inverse_distances[chosen[index]], baseline) for index, (_, patches) in enumerate(tiles)]
        counts = np.sum(fixed, axis=0).astype(int)
        chosen, determined = chosen.copy(), np.zeros(len(tiles), dtype=bool)
        order = [index for index in np.argsort([-len(patches) for _, patches in tiles], kind="stable") if refine[index]]
        for _ in range(self.n_sweeps):
            for index in order:
                others = counts - moved[index][chosen[index]]
                full = (others[None, :] > 0) | moved[index]
                window_weights = self._solid_angles * self.dilated(moved[index].any(axis=0))
                agreement = (full == target_obstructed[None, :]) @ window_weights / window_weights.sum()
                new = self.farthest_best(agreement)
                counts += moved[index][new].astype(int) - moved[index][chosen[index]]
                chosen[index], determined[index] = new, agreement.max() - agreement.min() > self.score_tolerance
        return chosen, determined

    def units(self, source_index):
        """Regions (or their azimuth sectors) of a group's sky map with their inverse distance, fitted against all other groups; cached.

        First each region as a whole: the farthest candidate within the tolerance of the best intersection over
        union; a region matches if the best reaches `min_score`, else it stays in place. Matched regions spanning
        several azimuth sectors (typically the horizon profile with buildings on it) are then split into sectors,
        refined together starting from their region's distance; sectors whose agreement doesn't depend on the
        distance keep their region's. With several other groups, the median is taken. Each unit is a dict with the patch indices, the region number, the inverse distance (0:
        infinitely far, or stays in place), whether it matched, and the best score per other group.
        """
        if source_index in self._units:
            return self._units[source_index]
        source = self.groups[source_index]
        targets = [target for target in self.groups if target is not source]
        regions = obstruction_regions(source.obstructed_sky)
        region_fits, region_scores = [[] for _ in regions], [{} for _ in regions]
        for target in targets:
            for number, region in enumerate(regions):
                scores = self.region_scores(region, target.offset_m - source.offset_m, target.obstructed_sky.obstructed)
                region_scores[number][target.name] = round(float(scores.max()), 3)
                if scores.max() >= self.min_score:
                    region_fits[number].append(self.farthest_best(scores))
        region_candidate = [int(np.median(fit)) if fit else 0 for fit in region_fits]
        ### Sectors of the matched regions spanning several; other regions stay one unit.
        tiles, refine, tile_pairs = [], [], []
        tile_of_patch = sector_index(self._centers, self.sector_width_deg) if self.sector_width_deg else np.zeros(self.sky.n_patches, dtype=int)
        for number, region in enumerate(regions):
            region_tiles = np.unique(tile_of_patch[region])
            split = bool(region_fits[number]) and len(region_tiles) > 1
            for tile in (region_tiles if split else [None]):
                tiles.append((number, region if tile is None else region[tile_of_patch[region] == tile]))
                refine.append(split)
        chosen = np.array([region_candidate[number] for number, _ in tiles], dtype=int)
        refined = []
        if any(refine):
            for target in targets:
                target_chosen, determined = self.refine_tiles(tiles, chosen, np.array(refine), target.offset_m - source.offset_m, target.obstructed_sky.obstructed)
                refined.append(np.where(determined, target_chosen, -1))
        units = []
        for index, (number, patches) in enumerate(tiles):
            candidates = [fit[index] for fit in refined if refine[index] and fit[index] >= 0]
            candidate = int(np.median(candidates)) if candidates else chosen[index]
            units.append({"patches": patches, "region": number, "inverse_distance": float(self.inverse_distances[candidate]), "matched": bool(region_fits[number]), "scores": region_scores[number]})
        self._units[source_index] = units
        log.debug(f"Photo group {source.name}: {len(regions)} regions ({sum(bool(fit) for fit in region_fits)} matched), {len(units)} units")
        return units

    def nearest_group_index(self, panel_position_m):
        """Index of the group nearest to the panel position (the first of equally near ones)."""
        return int(np.argmin([np.linalg.norm(group.offset_m - panel_position_m) for group in self.groups]))

    def region_summary(self, source_index, panel_position_m=(0.0, 0.0, 0.0)):
        """Per region or sector of the source group: size, centre direction at the panel position (azimuth, elevation in degrees), distance in metres (None: infinitely far, or no match) and match scores."""
        source = self.groups[source_index]
        baseline = np.asarray(panel_position_m, dtype=float) - source.offset_m
        summary = []
        for unit in self.units(source_index):
            centre = self._centers[unit["patches"]].T @ self._solid_angles[unit["patches"]]
            zenith, azimuth = zenith_azimuth_from_directions(moved_directions(centre / np.linalg.norm(centre), unit["inverse_distance"], baseline))
            summary.append({"region": unit["region"], "n_patches": int(len(unit["patches"])), "azimuth_deg": round(float(np.degrees(azimuth)), 1), "elevation_deg": round(float(90.0 - np.degrees(zenith)), 1),
                            "distance_m": round(1.0 / unit["inverse_distance"], 2) if unit["inverse_distance"] > 0 else None, "matched": unit["matched"], "scores": {name: round(score, 3) for name, score in unit["scores"].items()}})
        return summary

    def obstructed_flags(self, panel_position_m=(0.0, 0.0, 0.0), source_index=None):
        """Obstructed flag per patch at the panel position: the union of the source group's regions moved there (source: the nearest group unless given)."""
        panel_position_m = np.asarray(panel_position_m, dtype=float)
        source_index = self.nearest_group_index(panel_position_m) if source_index is None else source_index
        baseline = panel_position_m - self.groups[source_index].offset_m
        obstructed = np.zeros(self.sky.n_patches, dtype=bool)
        for unit in self.units(source_index):
            obstructed |= self.moved_region(unit["patches"], unit["inverse_distance"], baseline)
        return obstructed

    def obstructed_sky(self, panel_position_m=(0.0, 0.0, 0.0)):
        """Obstructed sky description at the panel position (east, north, up in metres relative to group A), with the groups, the position and the distance per region in the metadata."""
        panel_position_m = np.asarray(list(panel_position_m) + [0.0] * (3 - len(panel_position_m)), dtype=float)
        source_index = self.nearest_group_index(panel_position_m)
        source = self.groups[source_index]
        obstructed = self.obstructed_flags(panel_position_m, source_index) if len(self.groups) > 1 else source.obstructed_sky.obstructed.copy()
        metadata = {key: value for key, value in source.obstructed_sky.metadata.items() if key not in ("edits",)}
        methods = [*source.obstructed_sky.contributing_methods(), "position_interpolation"]
        metadata.update(method="+".join(methods), methods=methods, position_interpolation={
            "panel_position_m": [round(float(value), 3) for value in panel_position_m], "source_group": source.name, "settings": self.settings,
            "groups": [{"name": group.name, "offset_m": [round(float(value), 3) for value in group.offset_m], "method": group.obstructed_sky.metadata.get("method", ""),
                        "n_obstructed": int(np.count_nonzero(group.obstructed_sky.obstructed))} for group in self.groups],
            "regions": self.region_summary(source_index, panel_position_m) if len(self.groups) > 1 else []})
        result = ObstructedSky(self.sky, obstructed, metadata)
        log.info(f"Obstructed sky at the panel position {np.round(panel_position_m, 2).tolist()} m from group {source.name} ({len(self.groups)} groups): {np.count_nonzero(obstructed)} of {self.sky.n_patches} patches obstructed, "
                 f"horizontal sky view factor {result.sky_view_factor():.4f} (group {source.name}: {source.obstructed_sky.sky_view_factor():.4f})")
        return result
