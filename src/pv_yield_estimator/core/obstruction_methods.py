"""Registry of the sky obstruction methods, selected by `sky_obstruction.method` in the config."""

from pv_yield_estimator.core.lidar import LidarSkyObstruction

### Methods by name; further methods (photo, #17) plug in here.
SKY_OBSTRUCTION_METHODS = {LidarSkyObstruction.method_name: LidarSkyObstruction}


def sky_obstruction_method(method="lidar", **kwargs):
    """Create the configured obstruction method; its settings come from the sub-section named like the method.

    Example config section: `sky_obstruction: {method: lidar, lidar: {scanner_heading_deg: 188.1, min_points: 10}}`.
    """
    if method not in SKY_OBSTRUCTION_METHODS:
        raise ValueError(f"Unknown sky obstruction method {method!r}; known: {', '.join(SKY_OBSTRUCTION_METHODS)}")
    return SKY_OBSTRUCTION_METHODS[method](**(kwargs.get(method) or {}))
