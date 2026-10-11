"""Synthetic 3D scenes for the tests of the obstruction at other positions (#54): boxes and a far horizon profile.

The exact obstructed sky at any position comes from casting a ray along each patch centre: a patch is obstructed if
its ray hits a box or runs below the horizon profile (far away, the same from every position). Coordinates in metres,
x east, y north, z up, relative to the position of group A.
"""

import numpy as np

from pv_yield_estimator.core.sky import ObstructedSky, zenith_azimuth_from_directions


class SyntheticScene:
    """Axis-aligned boxes ((x0, y0, z0), (x1, y1, z1)) and a far horizon profile (elevation in degrees as a function of the compass azimuth in degrees)."""

    def __init__(self, boxes=(), horizon_profile=None):
        self.boxes = [(np.asarray(low, dtype=float), np.asarray(high, dtype=float)) for low, high in boxes]
        self.horizon_profile = horizon_profile

    def obstructed_flags(self, sky, position):
        """Exact obstructed flag per patch of `sky`, seen from `position` (east, north, up)."""
        directions = sky.patch_centers()
        position = np.asarray(position, dtype=float)
        obstructed = np.zeros(len(directions), dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = 1.0 / directions
            for low, high in self.boxes:
                ### Slab method: the ray enters all three slabs before it leaves any of them.
                t_low, t_high = (low - position) * inverse, (high - position) * inverse
                t_enter = np.nanmax(np.minimum(t_low, t_high), axis=1)
                t_leave = np.nanmin(np.maximum(t_low, t_high), axis=1)
                obstructed |= (t_leave >= np.maximum(t_enter, 0.0))
        if self.horizon_profile is not None:
            zenith, azimuth = zenith_azimuth_from_directions(directions)
            obstructed |= 90.0 - np.degrees(zenith) < self.horizon_profile(np.degrees(azimuth))
        return obstructed

    def obstructed_sky(self, sky, position, name=""):
        """Exact obstructed sky description seen from `position`, as the photo method would give it."""
        return ObstructedSky(sky, self.obstructed_flags(sky, position), {"method": "photo", "name": name})


def balcony_scene():
    """A balcony scene: groups at chest height on a balcony running east-west (A at x = 0, B at x = 4 m), facing south.

    - balcony roof slab overhead (0.9 m above the cameras, reaching 1.8 m to the south), a side wall at the west end;
    - a tree (box 3 m wide) 8 m to the south-east, a long building 18 m to the south, a tall building 35 m to the south-west;
    - a far horizon profile (hills) at 2–6° elevation.
    """
    boxes = [
        ((-1.0, -1.8, 0.9), (5.0, 0.6, 1.2)),          ### roof slab over the balcony
        ((-1.6, -1.8, -3.0), (-1.3, 0.6, 1.2)),        ### side wall at the west end
        ((5.0, -9.5, -4.0), (8.0, -6.5, 5.0)),         ### tree
        ((-10.0, -24.0, -8.0), (20.0, -18.0, 6.0)),    ### long building across the street
        ((-40.0, -50.0, -8.0), (-25.0, -35.0, 30.0)),  ### tall building
    ]
    return SyntheticScene(boxes, horizon_profile=lambda azimuth_deg: 4.0 + 2.0 * np.sin(np.radians(2 * azimuth_deg)))


def courtyard_scene():
    """Buildings around a courtyard at 6–15 m and a far horizon: obstructions in all directions, groups 3 m apart."""
    boxes = [
        ((-15.0, 6.0, -2.0), (15.0, 12.0, 9.0)),       ### north
        ((-15.0, -14.0, -2.0), (15.0, -10.0, 4.0)),    ### south, low
        ((9.0, -10.0, -2.0), (14.0, 6.0, 12.0)),       ### east, tall
        ((-7.0, -3.0, -2.0), (-6.0, -2.0, 6.0)),       ### a pole-like chimney 6 m to the west
    ]
    return SyntheticScene(boxes, horizon_profile=lambda azimuth_deg: 3.0 + 0.0 * azimuth_deg)


def garage_scene():
    """A flat garage top, groups at its two ends (A at x = 0, B at x = 6 m), cameras 1.4 m above the roof.

    - the house wall 2.5 m to the north (8 m high above the cameras), ending at x = 4 m;
    - a neighbour's house 9 m to the east, a tree 7 m to the south-west, a row of houses 25 m to the south;
    - a far horizon profile (hills) at 1–5° elevation.
    """
    boxes = [
        ((-6.0, 2.5, -4.0), (4.0, 10.0, 8.0)),         ### house wall to the north
        ((15.0, -6.0, -4.0), (23.0, 6.0, 5.0)),        ### neighbour's house to the east
        ((-7.0, -8.0, 0.5), (-3.0, -4.0, 6.0)),        ### tree crown to the south-west
        ((-30.0, -33.0, -6.0), (30.0, -25.0, 7.0)),    ### row of houses to the south
    ]
    return SyntheticScene(boxes, horizon_profile=lambda azimuth_deg: 3.0 + 2.0 * np.cos(np.radians(azimuth_deg - 150.0)))
