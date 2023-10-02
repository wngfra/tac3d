from ssp.interface import Object
from ssp.maps import Spatial2D
from ssp.utils import linear_steps, random_object, circle_points, square_points

import numpy as np

np.random.seed(0)  # to ensure test reproducibility

max_x = 2
max_y = 2
res = 41
tol_x = 2 * max_x / res
tol_y = 2 * max_y / res

obj_names = ["A", "B"]

objects = []
for name in obj_names:
    objects.append(random_object(name, max_x, max_y))

ssp_map = Spatial2D(dim=512, decode_threshold=0.5)
ssp_map.build_grid(x_len=max_x, y_len=max_y, x_spaces=res, y_spaces=res)


def test_map_single_object():
    """Tests placing and querying a single object in a map."""
    obj_x = 0.5
    obj_y = 1.2

    obj = Object(name="A", color=[0, 0, 0], shape=[1])
    obj.location = np.asarray((obj_x, obj_y))

    objects = [obj]
    ssp_map.update_from_objs(objects, reset=True)

    # test decoding the top coordinates occupied by an object
    assert np.allclose(ssp_map.decode_top_coords(), obj.location)

    # test decoding all coordinates occupied by an object
    for coord in ssp_map.decode_all_coords():
        assert np.abs(coord[0] - obj_x) < tol_x
        assert np.abs(coord[1] - obj_y) < tol_y

    # test that coordinates can be used to extract objects from SSP
    decoded = ssp_map.query_coords(obj.x, obj.y)
    assert decoded == obj.name

    # test that object name can query the location of the object
    coords = ssp_map.query_object(obj.name)
    assert np.abs(coords[0] - obj_x) < tol_x
    assert np.abs(coords[1] - obj_y) < tol_y


def test_map_shift():
    """Tests shifts of single object and all objects by specified amount"""
    dx = 0.1
    dy = 0.1

    ssp_map.update_from_objs(objects, reset=True)

    # move each object seperately and test match to target location
    for obj in objects:
        old_coords = ssp_map.query_object(obj.name)
        ssp_map.shift_unique(obj.name, dx, dy)
        new_coords = ssp_map.query_object(obj.name)
        assert np.abs(new_coords[0] - old_coords[0] - dx) < tol_x
        assert np.abs(new_coords[1] - old_coords[1] - dy) < tol_x

    # move all objects by a displacement and test match to target location
    ssp_map.update_from_objs(objects, reset=True)
    old_coords = []
    for obj in objects:
        old_coords.append(ssp_map.query_object(obj.name))

    ssp_map.shift_global(dx, dy)

    new_coords = []
    for obj in objects:
        new_coords.append(ssp_map.query_object(obj.name))

    for obj, old, new in zip(objects, old_coords, new_coords):
        assert np.abs(new[0] - old[0] - dx) < tol_x
        assert np.abs(new[1] - old[1] - dy) < tol_x


def test_map_regions():
    """Tests region encodings and queries"""
    xs, ys = circle_points(radius=0.1, n_points=100, x_offset=1, y_offset=1)

    # test placing object inside and outside of region
    obj_a = Object(name="A", color=[0, 0, 0], shape=[1])
    obj_a.location = np.array([1, 1])

    obj_b = Object(name="B", color=[0, 0, 0], shape=[1])
    obj_b.location = np.array([0.1, 0.1])

    ssp_map.reset()
    ssp_map.add_object(obj_a)
    ssp_map.add_object(obj_b)

    # test that only item within the region is retrieved
    decoded = ssp_map.query_region(xs, ys)
    assert len(decoded) == 1  # check that only 1 item is decoded
    assert decoded[0] == "A"  # check that the item is correct

    length = 0.2
    xs, ys = square_points(length=length, n_points=4, x_offset=0, y_offset=0)
    ssp_map.reset()
    ssp_map.add_region(xs, ys)

    # test that only points with the region are decoded
    for x, y in ssp_map.decode_all_coords():
        assert x < length + tol_x
        assert y < length + tol_y


def test_map_trajectory_enc_dec():
    """Tests encoding and decoding a continous trajectory on the plane"""
    ssp_map = Spatial2D(dim=1024, decode_threshold=0.5)
    ssp_map.build_grid(x_len=max_x, y_len=max_y, x_spaces=res, y_spaces=res)

    # use abs as test function (maybe expand to suite of functions?)
    xs = ssp_map.xs
    ys = np.abs(xs)
    points = list(zip(xs, ys))

    cues = linear_steps(len(points), stepsize=1)
    enc = ssp_map.encode_trajectory(points=points, cues=cues)
    dec = ssp_map.decode_trajectory(enc, cues=cues)

    sims = [ssp_map.compute_heatmap(v) for v in dec]
    # now check that the decoding matches the initial function samples
    for i, sim in enumerate(sims[:-1]):
        inds = np.unravel_index(sim.argmax(), sim.shape)  # largest sim inds

        # map max indices back to x, y values from -y, x used for sim plots
        x = ssp_map.xs[inds[1]]
        y = ssp_map.ys[-inds[0]]

        # test that decoded x, y values equal encoded x, y values
        assert np.abs(x - xs[i]) < tol_x
        assert np.abs(y - ys[i]) < tol_y
