import numpy as np
import pandas as pd

from collections import namedtuple


ObjectDynamics = namedtuple("ObjectDynamics", ["name", "x", "y", "vx", "vy"])


class Trajectory:
    """Stores configuration information defining a trajectory through the state
    space encoded by an SSP map. The trajectory specifies the dynamics of one
    or more objects that move over time through in accordance with some number
    of differential equations. Currently, these differential equations are
    assumed to relate an object's position to time.

    Parameters:
    ----------
    t: float
        The total timespan of the trajectory in seconds.
    dt: float
        The timestep used to simulate the trajectory over this timespan.
    """

    def __init__(self, t, dt):

        self.t = t
        self.dt = dt

        self.timesteps = np.linspace(0, t, int(t / dt))
        self.object_specs = []
        self.coords_track = {}

    @property
    def object_names(self):
        return [s.name for s in self.object_specs]

    def add_object_spec(self, name, init_x, init_y, dxdt, dydt):
        """Add configuration for an object's motion to the trajectory"""
        vx = dxdt(self.timesteps)
        vy = dydt(self.timesteps)

        spec = ObjectDynamics(name, init_x, init_y, vx, vy)
        self.object_specs.append(spec)


class TrajectoryLog:
    """Stores a set of arrays corresponding to the true and decoded positions
    of a set of objects over the course a trajectory. The true positions are
    determined analytically from the differential equations that govern each
    object's dynamics, while the decoded positions are left empty so as to be
    updated on the fly as an SSP is transformed to realize the target dynamics
    by some means.

    Parameters:
    -----------
    n: int
        The number of simulations to log.
    trajectory: Trajectory instance
        Configuration specifying the ground truth dynamics of each object.
    """

    def __init__(self, n, trajectory):

        self.true_coords = {}
        self.pred_coords = {}
        self.indices = {idx: 0 for idx in range(n)}  # tracks array updating

        self.timesteps = trajectory.timesteps
        self.n = n

        for spec in trajectory.object_specs:
            # initialize empty coordinate arrays
            self.true_coords[spec.name] = np.empty((n, len(self.timesteps), 2))
            self.pred_coords[spec.name] = np.empty((n, len(self.timesteps), 2))
            self.indices[spec.name] = 0

            x, y = spec.x, spec.y
            # determine sequence of ground-truth coordinates
            for i, step in enumerate(self.timesteps):
                x += spec.vx[i] * trajectory.dt
                y += spec.vy[i] * trajectory.dt

                # make n copies of ground truth coordinate, since it's fixed
                self.true_coords[spec.name][:, i, 0] = x
                self.true_coords[spec.name][:, i, 1] = y

    @property
    def global_error(self):
        """Get average error over all object trajectories"""
        rmses = [self.object_error(n) for n in self.pred_coords]
        return np.mean(rmses)

    def object_error(self, object_name):
        """Get RMSE for trajectory of a particular object"""
        pred = self.pred_coords[object_name]
        true = self.true_coords[object_name]
        rmse = np.sqrt(np.sum(np.square(true - pred), axis=1))

        return rmse

    def update(self, n, ssp_map, ssp):
        """Add values to the predicted coordinate array for a given object"""
        index = self.indices[n]  # tracks array indexing over trials
        for name in self.pred_coords:
            coords = ssp_map.decode_ssp_coords(ssp, name=name)
            if coords is not None:
                x, y = coords
                self.pred_coords[name][n, index, :] = [x, y]
            else:
                # if encoded position is "lost" treat last point as prediction
                # this is not defined properly for the first element, but that
                # should not lead to an error given that we start with a good
                # encoding
                x, y = self.pred_coords[name][n, index - 1, :]
                self.pred_coords[name][n, index, :] = [x, y]

        self.indices[n] += 1

    def to_dataframe(self):
        """Create a Pandas dataframe from the logger for easy plotting"""
        samples = []
        for sim in range(self.n):
            for i, step in enumerate(self.timesteps):
                for name in self.pred_coords:
                    # do x coordinates
                    sample = {"t": step, "Object": name, "trial": sim}
                    sample["val"] = self.true_coords[name][sim, i, 0]
                    sample["Prediction"] = False
                    sample["axis"] = "X"
                    samples.append(sample)

                    sample = {"t": step, "Object": name, "trial": sim}
                    sample["val"] = self.pred_coords[name][sim, i, 0]
                    sample["Prediction"] = True
                    sample["axis"] = "X"
                    samples.append(sample)

                    # do y coordinates
                    sample = {"t": step, "Object": name, "trial": sim}
                    sample["val"] = self.true_coords[name][sim, i, 1]
                    sample["Prediction"] = False
                    sample["axis"] = "Y"
                    samples.append(sample)

                    sample = {"t": step, "Object": name, "trial": sim}
                    sample["val"] = self.pred_coords[name][sim, i, 1]
                    sample["Prediction"] = True
                    sample["axis"] = "Y"
                    samples.append(sample)

        return pd.DataFrame(samples)
