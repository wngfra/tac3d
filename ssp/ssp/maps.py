import string
import nengo_spa as spa
import numpy as np

from ssp.pointers import BaseVectors


class Spatial2D:
    """Handles SSP representations of two-dimensional planes with entities
    occupying locations seperated by a continuous distance metric. Instances of
    this class are used to maintain, update, and perform queries on an SSP
    representation of some underlying 2D space. Updates can be performed
    on the basis of API calls to an underlying MuJoCo simulation that
    defines the 'ground truth' of the 2D space the SSP representation is
    intended to represent. Alternatively, updates can be performed on the
    basis of a set of `Object` instances that define the 'ground truth' of the
    2D space. Cleanups can be used to avoid compounding error with noisy
    decoding. Cleanup failures (no items over threshold) always return the zero
    vector, and ways of handling this in coordinate space is typically
    implemented.

    Parameters:
    ----------
    dim: int
        The dimensionality of the SSP representation of the plane.
    scale: int (optional)
        The degree "similarity spread" for a point SSP (higher -> more local).
    decode_threshold: float (optional)
        The noise threshold for implementing cleanup operations during
        decoding.
    unitary: bool (optional)
        Whether to use unitary vectors for the SSPs.
    X: nengo_spa.SemanticPointer (optional)
        Base vector to use for X axis.
    Y: nengo_spa.SemanticPointer (optional)
        Base vector to use for Y axis.
    rng: np.random.RandomState
        RNG state to use for semantic pointer generation.
    """

    def __init__(
        self,
        dim,
        scale=10,
        decode_threshold=0.5,
        unitary=True,
        X=None,
        Y=None,
        rng=np.random.RandomState(),
    ):

        self.dim = dim
        self.voc = spa.Vocabulary(self.dim, pointer_gen=BaseVectors(self.dim, rng=rng))
        self.scale = scale
        self.decode_threshold = decode_threshold
        self.unitary = unitary  # note: only applies to axis representations
        self.rng = rng

        # initial object names, map axes and trajectory encoding cue
        for key in [k for k in string.ascii_uppercase] + ["CUE"]:
            self.voc.populate(key + ".unitary()" if unitary else "")

        # TODO: should be a cleaner way to overwrite these base vectors
        if X is not None:
            self.voc._vectors[self.voc._key2idx["X"]] = X.v
        if Y is not None:
            self.voc._vectors[self.voc._key2idx["Y"]] = Y.v

        self.X = self.voc["X"]
        self.Y = self.voc["Y"]
        self.C = self.voc["CUE"]

        self.object_keys = [k for k in string.ascii_uppercase[:20]]

    @property
    def heatmap_scores(self):
        return self.compute_heatmap(self.ssp_mem)

    def compute_heatmap(self, ssp, names=None):
        """Compute heatmap sims using a provided SSP and grid points.

        If names are provided, then superimpose the heatmaps obtained
        from separately unbinding each name (without cleanup).
        """
        if names is not None:
            return np.sum(
                [self.compute_heatmap(ssp * ~self.voc[name]) for name in names], axis=0,
            )
        # switch from (x, y) to (-y, x) indexing to match plot origin
        # sim_tensor = np.swapaxes(self.ssp_tensor, 0, 1)
        # sim_tensor = np.flip(sim_tensor, axis=0)
        sim_tensor = self.ssp_tensor
        sims = np.tensordot(sim_tensor, ssp.v, axes=[[2], [0]])

        return sims

    def encode_point(self, x, y, name=None):
        """Create an optionally tagged SSP encoding of supplied coordinates"""
        tag = self.voc["Identity"] if name is None else self.voc[name]
        enc = tag * self.X ** (x * self.scale) * self.Y ** (y * self.scale)

        return enc

    def encode_points(self, xs, ys, names=None):
        """Create an optionally tagged SSP encoding of supplied coordinates."""
        if names is None:
            names = [None] * len(xs)  # assumes len(xs) == len(ys)
        return np.sum(
            [self.encode_point(x, y, name) for x, y, name in zip(xs, ys, names)]
        )

    def encode_region(self, xs, ys):
        """Create an SSP encoding that integrates over points in region."""
        # TODO(arvoelke): deprecate or replace with specific kinds of regions?
        return self.encode_points(xs, ys)

    def initialize_ssp(self, trajectory):
        """Create SSP that encodes initial positions in trajectory"""
        ssp = self.voc["Zero"]
        for spec in trajectory.object_specs:
            ssp += self.encode_point(spec.x, spec.y, spec.name)

        return ssp

    def initialize_cue(self, trajectory):
        """Create cue that decodes positions from SSP in trajectory"""
        cue = self.voc["Zero"]
        for spec in trajectory.object_specs:
            cue += ~self.voc[spec.name]

        return cue

    def reset(self):
        """Reset the stored SSP memory to the zero vector"""
        self.ssp_mem = self.voc["Zero"]

    def build_grid(self, x_len, y_len, x_spaces, y_spaces, centered=False):
        """Build table of SSP representations for making heatmaps."""
        self.x_len = x_len
        self.y_len = y_len
        self.x_spaces = x_spaces
        self.y_spaces = y_spaces

        self.xs = np.linspace(-self.x_len if centered else 0, self.x_len, self.x_spaces)
        self.ys = np.linspace(-self.y_len if centered else 0, self.y_len, self.y_spaces)

        # store grid of SSPs as a coordinate lookup table and as a tensor
        self.ssp_lookup = {}
        self.ssp_tensor = np.empty((len(self.xs), len(self.ys), self.dim))

        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                ssp = self.encode_point(x, y)
                self.ssp_lookup[(x, y)] = ssp
                self.ssp_tensor[i, j, :] = ssp.v

    def update_from_sim(self, interface, reset=True):
        """Create an SSP encoding of the current interface scene state."""
        if reset:
            self.ssp_mem = self.voc["Zero"]

        for obj in interface.object_lookup.values():
            # ensure SP for every object is in vocab
            if obj.name not in self.voc:
                self.voc.populate(obj.name)

            x, y, z = interface.get_xyz(obj.name)
            point_enc = self.encode_point(x, y)  # incl. position reps alone
            self.ssp_mem += point_enc + self.voc[obj.name] * point_enc

        return self.ssp_mem

    def update_from_objs(self, obj_set, reset=True):
        """Create an SSP encoding of the supplied set of objects."""
        if reset:
            self.ssp_mem = self.voc["Zero"]

        for obj in obj_set:
            self.add_object(obj)

        return self.ssp_mem

    def add_object(self, obj):
        """Add SSP encoding of the supplied object"""
        if obj.name not in self.voc:
            self.voc.populate(obj.name)

        point_enc = self.encode_point(obj.x, obj.y)  # incl. position alone
        self.ssp_mem += point_enc + self.voc[obj.name] * point_enc

    def add_point(self, x, y, reset=False):
        """Add SSPencoding of the supplied coordinate"""
        self.ssp_mem += self.encode_point(x, y)
    
    def add_region(self, xs, ys, reset=False):
        """Add SSP encoding that covers region tiled by xs and ys"""
        self.ssp_mem += self.encode_region(xs, ys)

    def cleanup(self, ssp, names, top_only=True, average=False):
        """Extract and cleanup all locations encoded by the SSP.

        Note that the returned pointer does not include any of the names.
        It is simply the superposition of X**x * Y**y for each object,
        where each object's position vector is cleaned up separately.
        """
        decoded = self.voc["Zero"]
        for name in names:
            coords = self.decode_ssp_coords(ssp, name, top_only, average)
            if coords is not None:
                x, y = coords
                decoded += self.encode_point(x, y)  # note: no name passed in

        return decoded

    def cleanup_by_name(self, ssp, name):
        """Decode position SSP corresponding to particular object name"""
        coords = self.decode_ssp_coords(ssp, name)
        if coords is not None:
            x, y = coords
            if name == 'Loc1':
                print('Decoded: ', x, y)
            return self.encode_point(x, y)
        else:
            return self.voc["Zero"]

    def decode_ssp_coords(self, ssp, name=None, top_only=True, average=False):
        """Decode coordinates represented in SSP using comparisons over grid"""
        if name:
            ssp = ssp * ~self.voc[name]  # extract coords of named object

        sims = np.tensordot(self.ssp_tensor, ssp.v, axes=[[2], [0]])
        if top_only:
            # this approach has the disadvantage of being slightly less
            # accurate, but the advantage of not being sensitive to the norm
            # of the vector being decoded since other items being above
            # threshold has no effect
            inds = np.unravel_index(sims.argmax(), sims.shape)
            if sims.flatten()[sims.argmax()] < self.decode_threshold:
                return None
            else:
                # convert grid indices back to corresponding spatial value
                x = self.xs[inds[0]]
                y = self.ys[inds[1]]
                return np.array([x, y])
        else:
            # this approach has the disadvantage of being sensitive to the
            # norm of the vector being decoded, but the advantage of being
            # slightly more accurate since it averages over decoded coordinates
            x_inds, y_inds = np.where(sims > self.decode_threshold)

            all_coords = []
            for x_ind, y_ind in zip(x_inds, y_inds):
                all_coords.append([self.xs[x_ind], self.ys[y_ind]])

            if not len(all_coords):
                return None

            # return average over all decoded coordinates
            stacked = np.vstack(all_coords)
            return np.mean(stacked, axis=0) if average else stacked

    def decode_top_coords(self):
        """Identify single location best encoded in the current SSP memory."""
        return self.decode_ssp_coords(self.ssp_mem)

    def decode_all_coords(self):
        """Identify all encoded locations as determined by threshold."""
        return self.decode_ssp_coords(self.ssp_mem, top_only=False)

    def query_object(self, name):
        """Get locations of object in current SSP memory representation."""
        decoding = self.ssp_mem * ~self.voc[name]
        loc_sims = np.tensordot(self.ssp_tensor, decoding.v, axes=[[2], [0]])

        x_inds, y_inds = np.where(loc_sims > self.decode_threshold)

        all_coords = []
        for x_ind, y_ind in zip(x_inds, y_inds):
            all_coords.append([self.xs[x_ind], self.ys[y_ind]])

        # average over represented grid points to get single cooridinate pair
        return np.mean(np.vstack(all_coords), axis=0)

    def query_coords(self, x, y):
        """Query what object is present at the specified coordinates."""
        loc_sp = self.ssp_mem * ~self.encode_point(x, y)
        scores = self.voc.dot(loc_sp)

        if max(scores) < self.decode_threshold:
            return "Empty Location"
        else:
            return self.voc._keys[np.argmax(scores)]

    def query_region(self, xs, ys):
        """Get objects contained in region defined by set of x and y coords"""
        region_query = self.encode_region(xs, ys).normalized()

        result = self.ssp_mem * ~region_query
        scores = self.voc.dot(result)
        keyids = np.where(scores > self.decode_threshold)[0]

        objects = [self.voc._keys[x] for x in keyids]
        return objects

    def shift_unique(self, name, dx, dy):
        """Move the specified object by specified increments along each axis"""
        old_x, old_y = self.query_object(name)
        new_x, new_y = old_x + dx, old_y + dy

        old_enc = self.encode_point(old_x, old_y)
        new_enc = self.encode_point(new_x, new_y)
        transform_sp = new_enc - old_enc

        self.ssp_mem += self.voc[name] * transform_sp

    def shift_global(self, dx, dy):
        """Move all objects by specified increments along each axis"""
        self.ssp_mem *= self.encode_point(dx, dy)

    def encode_trajectory(self, points, cues=None):
        """Create SSP that encodes a continuous trajectory over the 2D space"""
        encoding = self.voc["Zero"]

        # encode point sequentially along the cue axis defined by C
        for c, (x, y) in zip(cues, points):
            encoding += self.encode_point(x, y) * self.C ** c

        return encoding

    def decode_trajectory(self, trajectory, cues=None):
        """Decode points illustrating currently encoded trajectory"""
        decodings = []

        # now extract the point in the trajectory associated with each cue
        for c in cues:
            decoding = trajectory * self.C ** -c
            decodings.append(decoding)

        return decodings

    def simple_dynamics_gen(self, t, dt, dxdt, dydt):
        """Yield transformation SSPs specified in terms of velocities"""
        steps = np.linspace(0, t, int(t / dt))

        # provide derivative of position at each timestep to specify dynamics
        vel_xs = dxdt(steps)
        vel_ys = dydt(steps)

        for i, step in enumerate(steps):
            dX = self.X ** (vel_xs[i] * dt)
            dY = self.Y ** (vel_ys[i] * dt)

            dynamics_ssp = dX * dY
            yield dynamics_ssp

    def complex_dynamics_gen(self, t, dt, vx, vy, d2xdt, d2ydt):
        """Yield transformation SSPs specified in terms of accelerations"""
        steps = np.linspace(0, t, int(t / dt))

        # provide derivative of velocity at each timestep to specify dynamics
        acc_xs = d2xdt(steps)
        acc_ys = d2ydt(steps)

        for i, step in enumerate(steps):
            vx += acc_xs[i] * dt
            vy += acc_ys[i] * dt

            dX = self.X ** (vx * dt)
            dY = self.Y ** (vy * dt)

            dynamics_ssp = dX * dY
            yield dynamics_ssp

    def bounce_dynamics_gen(self, t, dt, ssp, vx, vy, d2xdt, d2ydt, rho=0.75):
        """Implement very simple elastic collision dynamics with solid floor"""
        steps = np.linspace(0, t, int(t / dt))

        # provide derivative of velocity at each timestep to specify dynamics
        acc_xs = d2xdt(steps)
        acc_ys = d2ydt(steps)

        # note we update the SSP in-place rather yield a transformation SSP
        # is necessary because each transformation depends on current SSP state
        for i, step in enumerate(steps):
            vx += acc_xs[i] * dt
            vy += acc_ys[i] * dt

            x, y = self.decode_ssp_coords(ssp)

            # determine floor position in terms of SSP grid map
            if y < self.y_len / self.y_spaces:
                vy = -rho * vy  # simple physics reverses and rescales velocity

            dX = self.X ** (vx * dt)
            dY = self.Y ** (vy * dt)

            ssp *= dX * dY
            yield ssp

    def algebraic_dynamics_gen(self, trajectory):
        """Simulate dynamics for multiple distinct objects with SSP algebra"""
        ssp = self.initialize_ssp(trajectory)
        for i, t in enumerate(trajectory.timesteps):
            for spec in trajectory.object_specs:
                dx = spec.vx[i] * trajectory.dt
                dy = spec.vy[i] * trajectory.dt

                old_enc = self.cleanup_by_name(ssp, spec.name)
                new_enc = old_enc * self.encode_point(dx, dy)

                ssp += self.voc[spec.name] * (new_enc - old_enc)

            yield ssp

    def modelled_dynamics_gen_v2(self, trajectory, model):
        """Simulate dynamics for multiple distinct objects with SSP velocity"""
        ssp = self.initialize_ssp(trajectory)
        for i, t in enumerate(trajectory.timesteps):
            vx = [spec.vx[i] * trajectory.dt for spec in trajectory.object_specs]
            vy = [spec.vy[i] * trajectory.dt for spec in trajectory.object_specs]
            vssp = self.encode_points(vx, vy, names=trajectory.object_names)
            out = model(np.hstack([ssp.v, vssp.v]))
            ssp = spa.SemanticPointer(out)
            yield ssp

    def modelled_dynamics_gen(self, n_steps, ssp, model):
        """Transform SSP over n_steps using a learned model for dynamics"""
        for step in range(n_steps):
            inp = ssp.v[np.newaxis, :]  # need 2D input for TF model
            out = np.squeeze(model(inp))
            ssp = spa.SemanticPointer(out)

            yield ssp
