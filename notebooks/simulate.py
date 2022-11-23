# Copyright 2022 wngfra.
# SPDX-License-Identifier: Apache-2.0

import pickle
from multiprocessing import Pool
from xml.etree import ElementTree as et

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control.utils import inverse_kinematics as ik
from ipywidgets import IntProgress
from IPython.display import display, HTML
from PIL import Image
from scipy.special import gamma
from sklearn.decomposition import PCA

# Rendering parameters
dpi = 100
framerate = 30  # (Hz)
width, height = 1280, 720

# IK solver parameters
_MAX_STEPS = 100
_TOL = 1e-12
_TIME_STEP = 2e-3  # Defined in XML. Default to 2e-3

# Define 3D arrays of balls connected with sliders
_M = (20, 20)
radius = 3e-4
mass = 1e-12 / np.prod(_M)
dx = (3e-2 - 2 * radius) / (_M[0] - 1)
offset = dx * (_M[0] - 1) / 2.0
geom_type = "sphere"

# Scene XML
robot_xml = "models/panda_nohand.xml"
scene_xml = "models/scene.xml"


def display_video(frames, framerate=framerate, figsize=None):
    try:
        height, width, _ = frames[0].shape
    except ValueError as e:
        height, width = frames[0].shape
    orig_backend = matplotlib.get_backend()
    # Switch to headless 'Agg' to inhibit figure rendering.
    matplotlib.use("Agg")
    if figsize is None:
        figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    return anim.to_html5_video()


def J_zero(physics, site_name):
    phys = mujoco.Physics.from_model(physics.model)
    jac_pos = np.zeros((3, phys.model.nv))
    jac_rot = np.zeros((3, phys.model.nv))
    mjlib.mj_jacSite(
        phys.model.ptr,
        phys.data.ptr,
        jac_pos,
        jac_rot,
        phys.model.name2id(site_name, "site"),
    )

    return np.vstack((jac_pos, jac_rot))


def mat2quat(mat):
    quat = np.empty(4)
    mjlib.mju_mat2Quat(quat, mat)
    return quat


def quatLocal(A, C):
    conjugateA = np.empty(4)
    B = np.empty(4)
    mjlib.mju_negQuat(conjugateA, mat2quat(A))
    mjlib.mju_mulQuat(B, conjugateA, mat2quat(C))
    return B


def reach_test(physics, site_name, target_name, joint_names, duration=2.0, rendered=True):
    w = 2*np.pi/duration  # Rotator angular speed
    omega = w*np.array([1, 1, 1])
    ctrl = np.empty(10)
    video = []
    stream = []
    orientations = []
    control_site = physics.data.site(name=site_name)
    target_site = physics.data.site(name=target_name)
    target_quat = mat2quat(control_site.xmat)
    target_xpos = target_site.xpos.copy()
    target_xpos[2] -= 1e-4
    smooth_factor = 0.5
    
    # Simulate, saving video frames
    physics.reset(0)
    physics.step()

    IP = IntProgress(min=0, max=int(duration / _TIME_STEP), description='Simulating {}:'.format(target_name))  # instantiate the progress bar
    display(IP)  # display the bar
    while physics.data.time < duration:
        IP.value += 1
        move_vec = (target_xpos - control_site.xpos)*min(smooth_factor, physics.data.time)/smooth_factor

        # Compute the inverse kinematics
        if np.linalg.norm(control_site.xpos - target_xpos) > _TOL:
            result = ik.qpos_from_site_pose(
                physics,
                site_name,
                target_pos=control_site.xpos + move_vec,
                target_quat=target_quat,
                joint_names=joint_names,
                tol=_TOL,
                max_steps=_MAX_STEPS,
                inplace=False,
            )
            ctrl[:7] = result.qpos[:7]
        ctrl[-3:] = omega*physics.data.time
        physics.set_control(ctrl)
        physics.step()
        
        # Save contact after 2s
        if len(physics.data.contact) > 0:
            pressure = physics.data.qpos[7:-3].copy().reshape(_M)
            stream.append(pressure)
            oris = physics.data.qpos[-3:].copy()
            orientations.append(oris)
            
        # Save video frames
        if rendered and len(video) < physics.data.time * framerate:
            pixels = physics.render(
                camera_id="prospective", width=width, height=height
            )
            video.append(pixels.copy())

    return video, np.asarray(stream), np.asarray(orientations)


# Load scene and define simulation variables
physics = mujoco.Physics.from_xml_path(scene_xml)
site_name = "attachment_site"
reach_sites = ["reach_site" + str(i + 1) for i in range(3)]
joint_names = ["joint{}".format(i + 1) for i in range(7)]

def sim_process(target_name):
    return reach_test(physics, site_name, target_name, joint_names, duration=5.0, rendered=True)

dataset = {}
with Pool(processes=3) as pool:
    results = pool.map(sim_process, reach_sites)