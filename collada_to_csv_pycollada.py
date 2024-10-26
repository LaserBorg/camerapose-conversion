import open3d as o3d
import collada
import numpy as np
import csv
from scipy.spatial.transform import Rotation as R


def find_cameras(node, parent_transform=np.identity(4)):
    # recursively find camera nodes and their transformations
    cameras = []
    if isinstance(node, collada.scene.Node):
        current_transform = parent_transform @ node.matrix
        for child in node.children:
            cameras.extend(find_cameras(child, current_transform))
    elif isinstance(node, collada.scene.CameraNode):
        cameras.append((node, parent_transform))
    return cameras

def extract_pointcloud(dae):
    # Assuming vertex_list is already populated with 3D vertices
    vertex_list = []
    for geometry in dae.geometries:
        for primitive in geometry.primitives:
            if isinstance(primitive, collada.triangleset.TriangleSet):
                for vertex in primitive.vertex:
                    vertex_list.append(vertex)

    vertices = np.array(vertex_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    return pcd

def extract_camera_params(dae):
    # Extract camera positions and orientations
    camera_info = []
    for scene in dae.scenes:
        for node in scene.nodes:
            cameras = find_cameras(node)
            for camera_node, transform in cameras:
                position = transform[:3, 3]
                orientation = transform[:3, :3]
                euler_angles = R.from_matrix(orientation).as_euler('xyz', degrees=True)
                camera_info.append({
                    'name': str(node.xmlnode.attrib.get('name', 'Unnamed')),
                    'position': position,
                    'euler_angles': euler_angles
                })
    return camera_info

def get_up_axis(dae):
    if dae.assetInfo.upaxis is not None:
        return dae.assetInfo.upaxis
    return "Y_UP"  # Default to Y_UP if not specified

def write_camera_params_to_csv(cam_params_list, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['#name', 'x', 'y', 'alt', 'heading', 'pitch', 'roll', 'f', 'px', 'py', 'k1', 'k2', 'k3', 'k4', 't1', 't2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for camera in cam_params_list:
            writer.writerow({
                '#name': camera['name'],
                'x': camera['position'][0],
                'y': camera['position'][1],
                'alt': camera['position'][2],
                'heading': round(camera['euler_angles'][0], 8),
                'pitch': round(camera['euler_angles'][1], 8),
                'roll': round(camera['euler_angles'][2], 8),
                'f': 0,  # Placeholder for focal length
                'px': 0,  # Principal point offset x
                'py': 0,  # Principal point offset y
                'k1': 0,  # Distortion coefficient k1
                'k2': 0,  # Distortion coefficient k2
                'k3': 0,  # Distortion coefficient k3
                'k4': 0,  # Distortion coefficient k4
                't1': 0,  # Tangential distortion coefficient t1
                't2': 0   # Tangential distortion coefficient t2
            })


if __name__ == '__main__':

    filename = 'assets/scene.dae'
    dae = collada.Collada(filename, ignore=[collada.common.DaeUnsupportedError, collada.common.DaeBrokenRefError])

    # Extract and save point cloud
    pcd = extract_pointcloud(dae)
    o3d.io.write_point_cloud("output/pointcloud.ply", pcd, write_ascii=False)

    # Extract camera intrinsics and extrinsics
    cam_params_list = extract_camera_params(dae)

    # Print camera information
    for info in cam_params_list:
        print(f"Cam: {info['name']}, Position: {info['position']}, Euler Angles: {info['euler_angles']}")

    # Write camera parameters to CSV
    write_camera_params_to_csv(cam_params_list, 'output/cameras_v2.csv')

    # Determine and print the up axis
    up_axis = get_up_axis(dae)
    print(f"Up Axis: {up_axis}")

    # Visualize
    o3d.visualization.draw_geometries([pcd])
