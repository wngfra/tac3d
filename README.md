# Tactile 3D Exploration and Perception Project
![Cover](./docs/cover.gif "Robot Touch")
![Sensor](./docs/softbody.png "Softbody Touch Sensor")
## Contents
1. `arduino_readout` contains the Arduino readout scheme for event-based tactile sensors
2. `CAD` contains the CAD files (AutoDesk Inventor)
3. `tacsense` contains the notebooks for tactile encoding with MuJoCo simulation
4. [SpikyMotion](https://github.com/wngfra/SpikyMotion) contains the ROS2 server for neuromorphic robotic control
5. `tactile_sensor` contains the KiCAD project of the sensor design
6. `loihi` contains the demos for Loihi with Nengo

## Notes
* Create a conda env with
  ```bash
    conda env create -n tac3d -f environment.yml
  ```
* Intel Lohi KapohoBay support(beta) is provided in the container, copy NxSDK tarballs to [docker](./loihi/docker/) directory, bring up the container in [loihi](./loihi/) with
  ```bash
    sudo rmmod ftdi_sio # Remove FTDI interfaces from kernel
    docker-compose up loihi
  ```

## TODO
1. Extract shape function with bivariate Generalized-Gaussian Spatial Model (GGSM) in SNN
2. Spiking control commands for robot in ROS2
