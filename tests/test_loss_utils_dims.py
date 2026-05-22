import numpy as np
import xarray as xr

from trainer.loss_utils import align_target_dims


def test_align_target_dims_handles_mixed_level_and_surface_vars():
    lat = np.linspace(-90.0, 90.0, 3, dtype=np.float32)
    lon = np.linspace(0.0, 360.0, 4, endpoint=False, dtype=np.float32)
    level = np.array([500, 850], dtype=np.int32)

    prediction = xr.Dataset(
        {
            "surface": (("time", "batch", "lat", "lon"), np.zeros((1, 1, 3, 4))),
            "upper": (("time", "batch", "level", "lat", "lon"), np.zeros((1, 1, 2, 3, 4))),
        },
        coords={"time": [0], "batch": [0], "level": level, "lat": lat, "lon": lon},
    )
    targets = xr.Dataset(
        {
            "surface": (("batch", "time", "lat", "lon"), np.zeros((1, 1, 3, 4))),
            "upper": (("batch", "time", "level", "lat", "lon"), np.zeros((1, 1, 2, 3, 4))),
        },
        coords={"time": [0], "batch": [0], "level": level, "lat": lat, "lon": lon},
    )

    aligned = align_target_dims(prediction, targets)

    assert aligned["surface"].dims == prediction["surface"].dims
    assert aligned["upper"].dims == prediction["upper"].dims
