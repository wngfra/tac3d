"""Adapted from https://scipython.com/blog/two-dimensional-collisions/"""

from itertools import combinations

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation

from nengo.utils.progress import Progress, ProgressTracker


def generate_collision_data(sim, dt, ssp_map, names, position_offset, n_collisions):
    # TODO(arvoelke): refactor position_offset
    d = ssp_map.voc.dimensions
    X = np.empty((n_collisions, 2 * d), dtype=np.float64)
    Y = np.empty((n_collisions, d), dtype=np.float64)

    i = 0
    last_state = None

    # these are currently helpful for cleaning up velocity
    low = [np.inf, np.inf]
    high = [-np.inf, -np.inf]

    with ProgressTracker(
        True, Progress("Generating", "Generation", n_collisions)
    ) as progress_bar:

        while i < n_collisions:
            P = ssp_map.encode_points(
                sim.x + position_offset, sim.y + position_offset, names,
            )

            # sim.adavance will scale the velocity by dt
            V = ssp_map.encode_points(dt * sim.vx, dt * sim.vy, names)

            low[0] = min(low[0], dt * np.min(sim.vx))
            low[1] = min(low[1], dt * np.min(sim.vy))
            high[0] = max(high[0], dt * np.max(sim.vx))
            high[1] = max(high[1], dt * np.max(sim.vy))

            sim.advance(dt)

            this_state = V.v
            if last_state is not None and not np.allclose(this_state, last_state):
                # only keep the state transitions
                X[i, :d] = P.v
                X[i, d:] = last_state
                Y[i, :] = this_state
                i += 1
                progress_bar.total_progress.step()
            last_state = this_state

    return X, Y, (low, high)


class Particle:
    """A class representing a two-dimensional particle."""

    def __init__(self, x, y, vx, vy, radius=0.01, styles=None):
        """Initialize the particle's position, velocity, and radius.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Circle patch constructor.

        """

        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius

        self.styles = styles
        if not self.styles:
            # Default circle styles
            self.styles = {"edgecolor": "b", "fill": False}

    # For convenience, map the components of the particle's position and
    # velocity vector onto the attributes x, y, vx and vy.
    @property
    def x(self):
        return self.r[0]

    @x.setter
    def x(self, value):
        self.r[0] = value

    @property
    def y(self):
        return self.r[1]

    @y.setter
    def y(self, value):
        self.r[1] = value

    @property
    def vx(self):
        return self.v[0]

    @vx.setter
    def vx(self, value):
        self.v[0] = value

    @property
    def vy(self):
        return self.v[1]

    @vy.setter
    def vy(self, value):
        self.v[1] = value

    def overlaps(self, other):
        """Does the circle of this Particle overlap that of other?"""

        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    def draw(self, ax):
        """Add this Particle's Circle patch to the Matplotlib Axes ax."""

        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle

    def advance(self, dt):
        """Advance the Particle's position forward in time by dt."""

        self.r += self.v * dt

        # Make the Particles bounce off the walls
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
        if self.x + self.radius > 1:
            self.x = 1 - self.radius
            self.vx = -self.vx
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
        if self.y + self.radius > 1:
            self.y = 1 - self.radius
            self.vy = -self.vy


class Simulation:
    """A class for a simple hard-circle molecular dynamics simulation.

    The simulation is carried out on a square domain: 0 <= x < 1, 0 <= y < 1.

    """

    def __init__(
        self, n, radius=0.01, dt=0.01, styles=None, rng=np.random.RandomState()
    ):
        """Initialize the simulation with n Particles with radii radius.

        radius can be a single value or a sequence with n values.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Circle patch constructor when drawing
        the Particles.

        """
        self.dt = dt
        self.init_particles(n, radius, styles, rng=rng)

    @property
    def x(self):
        return np.asarray([p.x for p in self.particles])

    @property
    def y(self):
        return np.asarray([p.y for p in self.particles])

    @property
    def vx(self):
        return np.asarray([p.vx for p in self.particles])

    @property
    def vy(self):
        return np.asarray([p.vy for p in self.particles])

    def init_particles(self, n, radius, styles, rng):
        """Initialize the n Particles of the simulation.

        Positions and velocities are chosen randomly; radius can be a single
        value or a sequence with n values.

        """

        try:
            iterator = iter(radius)
            assert n == len(radius)
        except TypeError:
            # r isn't iterable: turn it into a generator that returns the
            # same value n times.
            def r_gen(n, radius):
                for i in range(n):
                    yield radius

            radius = r_gen(n, radius)

        self.n = n
        self.particles = []
        for i, rad in enumerate(radius):
            # Try to find a random initial position for this particle.
            while True:
                # Choose x, y so that the Particle is entirely inside the
                # domain of the simulation.
                x, y = rad + (1 - 2 * rad) * rng.random(2)
                # Choose a random velocity (within some reasonable range of
                # values) for the Particle.
                vr = 0.1 * rng.random() + 0.05
                vphi = 2 * np.pi * rng.random()
                vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
                particle = Particle(x, y, vx, vy, rad, styles)
                # Check that the Particle doesn't overlap one that's already
                # been placed.
                for p2 in self.particles:
                    if p2.overlaps(particle):
                        break
                else:
                    self.particles.append(particle)
                    break

    def handle_collisions(self):
        """Detect and handle any collisions between the Particles.

        When two Particles collide, they do so elastically: their velocities
        change such that both energy and momentum are conserved.

        """

        def change_velocities(p1, p2):
            """
            Particles p1 and p2 have collided elastically: update their
            velocities.

            """

            m1, m2 = p1.radius ** 2, p2.radius ** 2
            M = m1 + m2
            r1, r2 = p1.r, p2.r
            d = np.linalg.norm(r1 - r2) ** 2
            v1, v2 = p1.v, p2.v
            u1 = v1 - 2 * m2 / M * np.dot(v1 - v2, r1 - r2) / d * (r1 - r2)
            u2 = v2 - 2 * m1 / M * np.dot(v2 - v1, r2 - r1) / d * (r2 - r1)
            p1.v = u1
            p2.v = u2

        # We're going to need a sequence of all of the pairs of particles when
        # we are detecting collisions. combinations generates pairs of indexes
        # into the self.particles list of Particles on the fly.
        pairs = combinations(range(self.n), 2)
        for i, j in pairs:
            if self.particles[i].overlaps(self.particles[j]):
                change_velocities(self.particles[i], self.particles[j])

    def step(self):
        """Step the simulation by internal simulator dt."""
        for i, p in enumerate(self.particles):
            p.advance(self.dt)
        self.handle_collisions()

    def advance(self, dt):
        """Advance the simulation by dt."""
        if not np.allclose(int(dt / self.dt) * self.dt, dt):
            raise ValueError("dt must be divisible by %s" % (self.dt))
        while dt > 0:
            self.step()
            dt -= self.dt

    def do_animation(self, dt, frames=800, interval=2):
        """Set up and carry out the animation of the molecular dynamics.

        To save the animation as a MP4 movie, set save=True.
        """

        circles = []

        def init():
            for particle in self.particles:
                circles.append(particle.draw(self.ax))
            return circles

        def animate(i):
            self.advance(dt)
            for i, p in enumerate(self.particles):
                circles[i].center = p.r
            return circles

        fig, self.ax = plt.subplots()
        for s in ["top", "bottom", "left", "right"]:
            self.ax.spines[s].set_linewidth(2)
        self.ax.set_aspect("equal", "box")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=frames, interval=interval, blit=True
        )
        plt.close()

        return anim
