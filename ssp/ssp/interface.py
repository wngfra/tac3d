import logging
import random
import numpy as np

from ssp.coppeliasim_files import sim
from itertools import count


class SimulationError(Exception):
    """An error dealing with CoppeliaSim."""

    pass


class Object:
    """Encapsulates data concerning an environment object.

    These objects are intended to encode environment dynamics and interface
    with CoppeliaSim to ensure that API calls respect these dynamics.

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
        Integers passed to Lua child-script function in CoppeliaSim
    inputFloats : list of floats
        Floats passed to Lua child-script function in CoppeliaSim
    inputStrings : list of strings
        Strings passed to Lua child-script function in CoppeliaSim
    inputBuffer : bytearray
        Buffer passed to Lua child-script function in CoppeliaSim
    handle : None
        Handle of obj in CoppeliaSim once it has been built
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
        """List of ints passed to Lua function in CoppeliaSim."""
        return self.shape

    @property
    def inputFloats(self):
        """List of floats passed to Lua function in CoppeliaSim."""
        return self.size + self.color + list(self.location) + self.mass

    @property
    def inputStrings(self):
        """List of strings passed to Lua function in CoppeliaSim."""
        return [self.name]

    @property
    def inputBuffer(self):
        """Buffer passed to Lua function in CoppeliaSim (required by API)."""
        return bytearray()  # currently no use of bytearrays


class CoppeliaSim:
    """Provides an interface to a CoppeliaSim simulation.

    Objects can be added to the simulation and their positions can be
    retrieved while they move around in accordance with the dynamics imposed
    by the CoppeliaSim physics simulator.

    Note that this interface is current setup to assume use with a specific
    CoppeliaSim scene file (`blocks.ttt`) supplied with this code.

    Parameters
    ----------
    x_len: int (optional)
        The length of the simulation environment along the x-axis in
        CoppeliaSim.
    y_len: int (optional)
        The length of the simulation environment along the y-axis in
        CoppeliaSim.
    """

    def __init__(self, x_len=2, y_len=2, *args, **kwargs):

        self.x_len = x_len  # init defaults are for `blocks.ttt`
        self.y_len = y_len

        # create grid of points to select from when initializing objects
        self.xs = np.linspace(0, self.x_len, 11)
        self.ys = np.linspace(0, self.y_len, 11)

        self.grid = [np.array([x, y]) for x in self.xs for y in self.ys]
        self.object_lookup = {}
        self.connected = False
        self._connect_params = (args, kwargs)

    def __enter__(self):
        self.connect(*self._connect_params[0], **self._connect_params[1])
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    @staticmethod
    def check_error_code(error_code):
        """Display CoppeliaSim error code if API call fails."""
        if error_code != 0:
            raise SimulationError("Call failure with code %d" % error_code)

    def connect(self, host="127.0.0.1", dt=0.005):
        """Connect to the current scene open in CoppeliaSim."""
        if self.connected:
            raise SimulationError("Cannot connect; already connected")

        # close any open connections
        sim.simxFinish(-1)
        # Connect to the CoppeliaSim continuous server
        self.clientID = sim.simxStart(host, 19997, True, True, 500, 5)

        if self.clientID == -1:
            raise SimulationError("Failed connecting to remote API server: %s" % host)

        sim.simxSynchronous(self.clientID, True)

        sim.simxSetFloatingParameter(
            self.clientID,
            sim.sim_floatparam_simulation_time_step,
            dt,  # specify a simulation time step
            sim.simx_opmode_oneshot,
        )

        sim.simxSetBooleanParameter(
            self.clientID,
            sim.sim_boolparam_display_enabled,
            True,
            sim.simx_opmode_blocking,
        )

        # start our simulation in lockstep with our code
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)

        logging.info("Connected to CoppeliaSim remote API server")
        self.connected = True

    def disconnect(self):
        """Stop and reset the simulation."""
        if not self.connected:
            raise SimulationError("Cannot disconnect; not connected")

        # stop the simulation
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)

        # Before closing the connection to CoppeliaSim,
        # make sure that the last command sent out had time to arrive.
        sim.simxGetPingTime(self.clientID)

        # Now close the connection to CoppeliaSim
        sim.simxFinish(self.clientID)
        logging.info("CoppeliaSim connection closed...")

    def get_handle(self, name):
        """Get the simulator handle associated with an object name."""
        if not self.connected:
            raise SimulationError("Cannot get handle; not connected")

        _, handle = sim.simxGetObjectHandle(
            self.clientID, name, sim.simx_opmode_blocking
        )

        return handle

    def get_xyz(self, name):
        """Get the current xyz coordinates of an object in CoppeliaSim."""
        handle = self.get_handle(name)

        _, xyz = sim.simxGetObjectPosition(
            self.clientID, handle, -1, sim.simx_opmode_blocking
        )

        return xyz

    def set_xyz(self, name, xyz):
        """Set the xyz coordinates of an object in CoppeliaSim."""
        handle = self.get_handle(name)

        sim.simxSetObjectPosition(
            self.clientID, handle, -1, xyz, sim.simx_opmode_blocking
        )

    def get_orientation(self, name, in_degrees=True):
        """Get the orientation of an object in CoppeliaSim."""
        handle = self.get_handle(name)

        _, angles = sim.simxGetObjectOrientation(
            self.clientID, handle, -1, sim.simx_opmode_blocking
        )

        angles = [np.rad2deg(x) for x in angles] if in_degrees else angles

        return angles

    def set_orientation(self, name, angles, in_degrees=True):
        """Set the orientation of an object in CoppeliaSim."""
        handle = self.get_handle(name)
        angles = [np.deg2rad(x) for x in angles] if in_degrees else angles

        sim.simxSetObjectOrientation(
            self.clientID, handle, -1, angles, sim.simx_opmode_blocking
        )

    def set_sensor(self, name):
        """Store sensor handle retrived name for convenient image capture"""
        self.sensor = sim.simxGetObjectHandle(
            self.clientID, name, sim.simx_opmode_oneshot_wait
        )

        self.sensor_data = []

    def call_build_script(self, obj):
        """Build an object into an open CoppeliaSim simulation."""
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
        """Add a randomly configured object to the CoppeliaSim scene."""
        color = np.random.uniform(0, 1, size=3)
        obj = Object(name=name, color=color, shape=[1])

        pos = random.choice(self.grid)
        pos = np.array([pos[0], pos[1], 0.05])
        obj.location = pos

        self.call_build_script(obj)
        self.object_lookup[name] = obj

    def delete_object(self, name):
        """Delete the named object from the CoppeliaSim scene."""
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
        """Advance the CoppeliaSim simulation by one time step (dt)."""
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
