import itertools
import numpy as np

from ssp.interface import Object
from ssp.pointers import normalize
from ssp.dynamics import Trajectory
from functools import partial


def linear_steps(n_steps, stepsize):
    """Create linearly spaced array with specified stepsize"""
    return stepsize * np.arange(n_steps)


def interpolate(array, n):
    """Linearly interpolate n elements between adjacent elements in 1D array"""
    segments = []

    for e1, e2 in zip(array, array[1:]):
        # make each segment start at e1 add n even steps without including e2
        segments.append(np.linspace(e1, e2, n + 2)[:-1])

    return np.hstack(segments)


def circle_points(radius, n_points, n_tilings=3, x_offset=0, y_offset=0):
    """Create points tiling a circle to integrate over in a region SSP"""
    tile_scaling = np.arange(0, n_tilings + 1)  # linearly add points going out

    n_per_tiling = np.rint(tile_scaling * n_points / np.sum(tile_scaling))
    n_per_tiling[0] = 1  # add a single point for the circle's center
    r_per_tiling = np.linspace(0, radius, n_tilings + 1)

    xs = []
    ys = []
    for r, n in zip(r_per_tiling, n_per_tiling):
        t = np.linspace(0, 2 * np.pi, int(n))
        xs.append(r * np.cos(t) + x_offset)
        ys.append(r * np.sin(t) + y_offset)

    xs = np.hstack(xs)
    ys = np.hstack(ys)

    return xs, ys


def square_points(length, n_points, x_offset=0, y_offset=0):
    """Create points tiling a square to integrate over in a region SSP"""
    n_per_tiling = int(np.sqrt(n_points))
    xs = []
    ys = []

    x_span = np.linspace(0, length, n_per_tiling)
    y_span = np.linspace(0, length, n_per_tiling)

    for x, y in itertools.product(x_span, y_span):
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def random_object(name, max_x, max_y):
    """Create a random object located within specified location extent."""
    color = np.random.uniform(0, 1, size=3)  # random RGB values
    obj = Object(name=name, color=color, shape=[1])  # spheres in CoppeliaSim

    x = np.random.uniform(0, max_x)
    y = np.random.uniform(0, max_y)
    z = 0.05  # fixed for now, sets objects on main surface in CoppeliaSim

    pos = np.array([x, y, z])
    obj.location = pos

    return obj


def create_data(ssp_map, trajectories, add_noise=False, include_velocity=False):
    """Make x, y pairs of SSPs corresponding to adjacent trajectory points"""
    xs, ys = [], []
    coords = {}

    for trajectory in trajectories:
        ssp = ssp_map.initialize_ssp(trajectory)

        for spec in trajectory.object_specs:
            coords[spec.name] = [spec.x, spec.y]

        for index, _ in enumerate(trajectory.timesteps):
            x_ssp = ssp.v

            if include_velocity:
                # TODO: this is duplicated across modelled_dynamics_gen_v2
                vx = [
                    spec.vx[index] * trajectory.dt for spec in trajectory.object_specs
                ]
                vy = [
                    spec.vy[index] * trajectory.dt for spec in trajectory.object_specs
                ]
                v_ssp = ssp_map.encode_points(vx, vy, names=trajectory.object_names).v

            # use ground truth data updated according to dx, dy
            for spec in trajectory.object_specs:
                x, y = coords[spec.name]

                dx = spec.vx[index] * trajectory.dt
                dy = spec.vy[index] * trajectory.dt

                old_enc = ssp_map.encode_point(x, y)
                new_enc = old_enc * ssp_map.encode_point(dx, dy)
                coords[spec.name] = [x + dx, y + dy]

                ssp += ssp_map.voc[spec.name] * (new_enc - old_enc)

            y_ssp = ssp.v

            if add_noise:
                # for each step make ten noisy variants of the x, y ssp pair
                for i in range(5):
                    x = x_ssp + 0.02 * normalize(np.random.randn(ssp_map.dim))
                    y = y_ssp + 0.02 * normalize(np.random.randn(ssp_map.dim))
                    if include_velocity:
                        v = v_ssp + 0.02 * normalize(np.random.randn(ssp_map.dim))
                        xs.append(np.concatenate((x, v)))
                    else:
                        xs.append(x)
                    ys.append(y)
            else:
                if include_velocity:
                    xs.append(np.concatenate((x_ssp, v_ssp)))
                else:
                    xs.append(x_ssp)
                ys.append(y_ssp)

    # perm = np.random.permutation(np.arange(len(xs)))
    xs = np.vstack(xs)  # [perm]
    ys = np.vstack(ys)  # [perm]

    return xs, ys


def oscillating_trajectory(ssp_map, t=5, dt=0.05, n_objects=5, noise=False):
    """Generate a test trajectory with 5 objects over specified timesteps"""
    trajectory = Trajectory(t, dt)
    # randomly perturb starting locations to get some variability
    if noise:
        shift_x, shift_y = 0.3 * np.random.uniform(-1, 1, size=2)
    else:
        shift_x, shift_y = 0, 0

    # either name each object seperately or give them all the same name
    # TODO: make dynamics methods on ssp_map name agnostic?
    osc_names = ["A", "B"]
    lin_names = ["C", "D"]
    signs = [-1, 1]

    # add two circular oscillations in opposing directions
    for sign, name in zip(signs, osc_names):
        dxdt = partial(lambda sign, t: sign * np.sin(t * np.pi), sign)
        dydt = partial(lambda sign, t: sign * np.cos(t * np.pi), sign)
        trajectory.add_object_spec(name, 1 + shift_x, 1 + shift_y, dxdt, dydt)

    # add two lines oscillating from left to right across the screen
    for sign, name in zip(signs, lin_names):
        init = 0.5 if sign == 1 else 1.5  # top vs bottom of scene
        dxdt = partial(lambda m, t: m.x_len * np.sin(t * np.pi), ssp_map)
        dydt = partial(lambda sign, t: sign * np.cos(t * 3 * np.pi), sign)
        trajectory.add_object_spec(name, 0.1, init + shift_y, dxdt, dydt)

    # add final sine wave moving up the middle of the scene
    dxdt = lambda t: np.sin(t * 5 * np.pi)  # noqa: E731
    dydt = lambda t: np.sin(t * 1 * np.pi)  # noqa: E731
    trajectory.add_object_spec("E", 1 + shift_x, 0.1, dxdt, dydt)

    trajectory.object_specs = trajectory.object_specs[:n_objects]

    return trajectory


def converging_trajectory(ssp_map, t=5, dt=0.05):
    """Generate a test trajectory with 5 objects over specified timesteps"""
    trajectory = Trajectory(t, dt)

    # either name each object seperately or give them all the same name
    # TODO: make dynamics methods on ssp_map name agnostic?
    osc_names = ["A", "B"]
    lin_names = ["C", "D"]
    signs = [-1, 1]

    # add two circular oscillations in opposing directions
    for sign, name in zip(signs, osc_names):
        dxdt = partial(lambda sign, t: sign * np.sin(t * np.pi), sign)
        dydt = partial(lambda sign, t: sign * np.cos(t * np.pi), sign)
        trajectory.add_object_spec(name, 1, 1, dxdt, dydt)

    # add two lines oscillating from left to right across the screen
    for sign, name in zip(signs, lin_names):
        init = 0 if sign == 1 else 2  # top vs bottom of scene
        dxdt = lambda t: 0.5 * np.ones(len(t))  # noqa: E731
        dydt = partial(lambda sign, t: 0.5 * sign * np.ones(len(t)), sign)
        trajectory.add_object_spec(name, 0, init, dxdt, dydt)

    # add final sine wave moving up the middle of the scene
    dxdt = lambda t: np.sin(t * 5 * np.pi)  # noqa: E731
    dydt = lambda t: 0.5 * np.ones(len(t))  # noqa: E731
    trajectory.add_object_spec("E", 1, 0, dxdt, dydt)

    return trajectory
