from ssp.maps import Spatial2D
from ssp.dynamics import Trajectory, TrajectoryLog

import numpy as np


max_x = 2
max_y = 2
res = 41
tol_x = 2 * max_x / res
tol_y = 2 * max_y / res

T = 2
dt = 0.05
n = 1


def test_logging():
    """Tests that logging can be used to compute RMSE on trajectories"""
    ssp_map = Spatial2D(dim=256, decode_threshold=0.2)
    ssp_map.build_grid(x_len=max_x, y_len=max_y, x_spaces=res, y_spaces=res)

    trajectory = Trajectory(T, dt)
    trajectory.add_object_spec(
        name="A",
        init_x=0,
        init_y=0,
        dxdt=lambda t: np.ones(len(t)),
        dydt=lambda t: np.ones(len(t)),
    )

    logger = TrajectoryLog(n, trajectory)

    for new_ssp in ssp_map.algebraic_dynamics_gen(trajectory):
        logger.update(0, ssp_map, new_ssp)

    assert logger.global_error <= tol_x + tol_y
    assert np.allclose(logger.object_error("A"), logger.global_error)
