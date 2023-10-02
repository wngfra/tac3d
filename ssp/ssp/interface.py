import logging
import random
import mujoco
import numpy as np

from itertools import count


class SimulationError(Exception):
    """An error dealing with MuJoCo."""

    pass


class Object:
    """Encapsulates data concerning an environment object.

    These objects are intended to encode environment dynamics and interface
    with MuJoCo to ensure that API calls respect these dynamics.

    Parameters
    ----------
    shape : array-like int
        The int code for type of object (cube, sphere, cylinder, or cone)
    color : array-like of floats
        The color of the object as an RGB value.

    Attributes
    ----------
    x : float
        The x-coordinate for the object's position.
    y : float
        The y-coordinate for the object's position.
    z : float
        The z-coordinate for the object's position.
    inputInts : list of ints
        Integers passed to Lua child-script function in MuJoCo
    inputFloats : list of floats
        Floats passed to Lua child-script function in MuJoCo
    inputStrings : list of strings
        Strings passed to Lua child-script function in MuJoCo
    inputBuffer : bytearray
        Buffer passed to Lua child-script function in MuJoCo
    handle : None
        Handle of obj in MuJoCo once it has been built
    """

    id_gen = count(0)  # tracks number of objects that have been created

    def __init__(self, name, color, shape):
        self.name = str(name)
        self.color = list(color)
        self.shape = list(shape)
        self.location = None  # only set if object is place in environment

        self.size = [0.1, 0.1, 0.1]
        self.mass = [1.0]

        self.sim_dummy = "ObjectGenerator"
        self.sim_function = "createObject_function"

        self._id = next(self.id_gen)

        self.handle = None

    def __str__(self):
        return str(self._id)

    def __eq__(self, other):
        return self._id == other._id

    def __hash__(self):
        return hash(self._id)

    @property
    def id(self):
        """Return the immutable ID of the object."""
        return self._id

    @property
    def x(self):
        """Return the x-coordinate of the object."""
        return float(self.location[0])

    @property
    def y(self):
        """Return the y-coordinate of the object."""
        return float(self.location[1])

    @property
    def z(self):
        """Return the z-coordinate of the object."""
        return float(self.location[2])

    @property
    def inputInts(self):
        """List of ints passed to Lua function in MuJoCo."""
        return self.shape

    @property
    def inputFloats(self):
        """List of floats passed to Lua function in MuJoCo."""
        return self.size + self.color + list(self.location) + self.mass

    @property
    def inputStrings(self):
        """List of strings passed to Lua function in MuJoCo."""
        return [self.name]

    @property
    def inputBuffer(self):
        """Buffer passed to Lua function in MuJoCo (required by API)."""
        return bytearray()  # currently no use of bytearrays


class MJCSim:
    """Provides an interface to a MuJoCo simulation.

    Objects can be added to the simulation and their positions can be
    retrieved while they move around in accordance with the dynamics imposed
    by the MuJoCo physics simulator.

    Parameters
    ----------
    x_len: int (optional)
        The length of the simulation environment along the x-axis in
        MuJoCo.
    y_len: int (optional)
        The length of the simulation environment along the y-axis in
        MuJoCo.
    """

    def __init__(self, x_len=2, y_len=2, *args, **kwargs):
        xml = """
        """
        model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(model)
        
        self.x_len = x_len
        self.y_len = y_len

        # create grid of points to select from when initializing objects
        self.xs = np.linspace(0, self.x_len, 11)
        self.ys = np.linspace(0, self.y_len, 11)

        self.grid = [np.array([x, y]) for x in self.xs for y in self.ys]
        self.object_lookup = {}

    def get_xyz(self, name):
        """Get the current xyz coordinates of an object in MuJoCo."""
        
        xyz = self.data.geom(name).xpos

        return xyz

    def set_xyz(self, name, xyz):
        """Set the xyz coordinates of an object in MuJoCo."""
        
        self.data.geom(name).xpos = xyz

    def get_orientation(self, name, in_degrees=True):
        """Get the orientation of an object in MuJoCo."""
        
        xmat = self.data.geom(name).xmat
        # TODO: convert to euler angles
        return angles

    # TODO: Implementation of the MuJoCo API.
    
    def set_orientation(self, name, angles, in_degrees=True):
        """Set the orientation of an object in MuJoCo."""
        angles = [np.deg2rad(x) for x in angles] if in_degrees else angles

    def set_sensor(self, name):
        """Store sensor handle retrived name for convenient image capture"""
        self.sensor = sim.simxGetObjectHandle(
            self.clientID, name, sim.simx_opmode_oneshot_wait
        )

        self.sensor_data = []

    def call_build_script(self, obj):
        """Build an object into an open MuJoCo simulation."""
        if not self.connected:
            raise SimulationError("Cannot call build script; not connected")

        error_code, ints, floats, strings, buff = sim.simxCallScriptFunction(
            clientID=self.clientID,
            scriptDescription=obj.sim_dummy,
            options=sim.sim_scripttype_childscript,
            functionName=obj.sim_function,
            inputInts=obj.inputInts,
            inputFloats=obj.inputFloats,
            inputStrings=obj.inputStrings,
            inputBuffer=obj.inputBuffer,
            operationMode=sim.simx_opmode_blocking,
        )

        self.check_error_code(error_code)
        obj.handle = ints[0]
        sim.simxSynchronousTrigger(self.clientID)

    def create_object(self, name):
        """Add a randomly configured object to the MuJoCo scene."""
        color = np.random.uniform(0, 1, size=3)
        obj = Object(name=name, color=color, shape=[1])

        pos = random.choice(self.grid)
        pos = np.array([pos[0], pos[1], 0.05])
        obj.location = pos

        self.call_build_script(obj)
        self.object_lookup[name] = obj

    def delete_object(self, name):
        """Delete the named object from the MuJoCo scene."""
        handle = self.get_handle(name)
        sim.simxRemoveObject(self.clientID, handle, sim.simx_opmode_blocking)

    def apply_force(self, name, position=None, force=None):
        """Apply a force to named object at relative position"""
        if not self.connected:
            raise SimulationError("Cannot apply force; not connected")

        obj = self.object_lookup[name]

        if position is None:
            position = np.random.randint(-1, 1, size=3)

        if force is None:
            force = np.random.uniform(-2, 2, size=3)

        error_code, ints, floats, strings, buff = sim.simxCallScriptFunction(
            clientID=self.clientID,
            scriptDescription=obj.sim_dummy,
            options=sim.sim_scripttype_childscript,
            functionName="applyForce_function",
            inputInts=list(position),
            inputFloats=list(force),
            inputStrings=obj.inputStrings,
            inputBuffer=obj.inputBuffer,
            operationMode=sim.simx_opmode_blocking,
        )

        self.check_error_code(error_code)
        sim.simxSynchronousTrigger(self.clientID)

    def advance(self):
        """Advance the MuJoCo simulation by one time step (dt)."""
        if not self.connected:
            raise SimulationError("Cannot advance simulation; not connected")

        if hasattr(self, "sensor"):
            err, res, image_data = sim.simxGetVisionSensorImage(
                self.clientID, self.sensor[1], 0, sim.simx_opmode_streaming
            )

            # first capture is often empty, hence this check
            if image_data != []:
                image = np.flipud(
                    np.reshape(np.asarray(image_data), (res[1], res[0], 3))
                )

                image = image.astype(np.uint8)
                self.sensor_data.append(image)

        sim.simxSynchronousTrigger(self.clientID)
