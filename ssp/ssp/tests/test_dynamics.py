from ssp.maps import Spatial2D
from ssp.dynamics import TrajectoryLog
from ssp.utils import converging_trajectory, oscillating_trajectory
from ssp.utils import create_data
from ssp.models import Linear

import numpy as np


dim = 512
max_x = 2
max_y = 2
res = 61
tol_x = 2 * max_x / res
tol_y = 2 * max_y / res

T = 2
dt = 0.05
n = 1
n_steps = int(T / dt)


def test_converging():
    """Tests that converging trajectory with 5 objects is below tolerance"""
    ssp_map = Spatial2D(dim=dim, decode_threshold=0.3)
    ssp_map.build_grid(x_len=max_x, y_len=max_y, x_spaces=res, y_spaces=res)

    trajectory = converging_trajectory(ssp_map, T, dt)
    logger = TrajectoryLog(n, trajectory)

    xs, ys = create_data(ssp_map, [trajectory])

    model = Linear(x_dim=dim, y_dim=dim)
    model.train(xs, ys, n_steps=200)

    ssp = ssp_map.initialize_ssp(trajectory)

    for new_ssp in ssp_map.modelled_dynamics_gen(n_steps, ssp, model):
        logger.update(0, ssp_map, new_ssp)

    for name in trajectory.object_names:
        assert np.mean(logger.object_error(name)) <= tol_x + tol_y


def test_oscillating():
    """Tests that converging trajectory with 5 objects is below tolerance"""
    ssp_map = Spatial2D(dim=dim, decode_threshold=0.3)
    ssp_map.build_grid(x_len=max_x, y_len=max_y, x_spaces=res, y_spaces=res)

    trajectory = oscillating_trajectory(ssp_map, T, dt)
    logger = TrajectoryLog(n, trajectory)

    xs, ys = create_data(ssp_map, [trajectory])

    model = Linear(x_dim=dim, y_dim=dim)
    model.train(xs, ys, n_steps=200)

    ssp = ssp_map.initialize_ssp(trajectory)

    for new_ssp in ssp_map.modelled_dynamics_gen(n_steps, ssp, model):
        logger.update(0, ssp_map, new_ssp)

    for name in trajectory.object_names:
        assert np.mean(logger.object_error(name)) <= tol_x + tol_y
