# Tactile 3D Exploration and Perception Project
![Cover](./docs/cover.gif "Robot Touch")
![Sensor](./docs/softbody.png "Softbody Touch Sensor")
## Contents
1. `arduino_readout` contains the Arduino readout scheme for event-based tactile sensors.
2. `CAD` contains the CAD files (AutoDesk Inventor).
3. `notebooks` contains the notebooks of tactile encoding study and MuJoCo simulation.
4. [SpikyMotion](https://github.com/wngfra/SpikyMotion) contains the ROS2 server for neuromorphic robotic control.
5. `sensor_design` contains the KiCAD project of the sensor design.
6. `tac3d` contains the [ROS2](https://docs.ros.org/en/humble/index.html) packages for active tactile exploration simulation.

## Notes
* Intel Lohi KapohoBay support(beta) is provided in the container, copy NxSDK tarballs to [docker](docker/) directory, setup the kernel before runing the container
  ```bash
    sudo rmmod ftdi_sio # Remove FTDI interfaces from kernel
  ```