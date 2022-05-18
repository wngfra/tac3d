import os
import requests
from urllib.parse import urljoin

base_url = "https://raw.githubusercontent.com/frankaemika/franka_ros/develop/franka_description/"
urdf_names = ["hand.urdf.xacro", "hand.xacro", "inertial.yaml",
              "panda_arm.urdf.xacro", "panda_arm.xacro", "utils.xacro"]
mesh_urls = ["meshes/collision/", "meshes/visual/"]
mesh_names = ["finger", "hand", "link0", "link1",
              "link2", "link3", "link4", "link5", "link6", "link7"]
mesh_exts = [".dae", ".stl"]

urdf_dir = "robots"
if not os.path.isdir(urdf_dir):
    os.mkdir(urdf_dir)

# Get urdf xacros
for urdf_name in urdf_names:
    url = urljoin(urljoin(base_url, "robots/"), urdf_name)
    res = requests.get(url)
    if res.status_code == 200:
        save_path = os.path.join(urdf_dir, urdf_name)
        with open(save_path, "wb") as f:
            f.write(res.content)

# Get meshes
for mesh_url in mesh_urls:
    if not os.path.isdir(mesh_url):
        os.makedirs(mesh_url)
    for mesh_name in mesh_names:
        for mesh_ext in mesh_exts:
            url = urljoin(urljoin(base_url, mesh_url), mesh_name+mesh_ext)
            res = requests.get(url)
            if res.status_code == 200:
                save_path = os.path.join(mesh_url, mesh_name+mesh_ext)
                with open(save_path, "wb") as f:
                    f.write(res.content)

print("Finished downloading files!")