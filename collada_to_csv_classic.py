import csv
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R


def convert_y_up_to_z_up(matrix):
    # Transformation matrix to convert Y_UP to Z_UP
    transform = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    return np.dot(transform, matrix)

def rotation_matrix_to_euler(matrix):
    # Ensure the matrix is a valid rotation matrix
    if not np.allclose(np.dot(matrix, matrix.T), np.eye(3), atol=1e-6):
        raise ValueError("The provided matrix is not a valid rotation matrix.")
    
    r = R.from_matrix(matrix)
    euler_angles = r.as_euler('xyz', degrees=True)
    return euler_angles

def calculate_focal_length(xfov, sensor_width_mm):
    # Calculate the focal length from the horizontal field of view (XFOV)
    focal_length = (sensor_width_mm / 2) / np.tan(np.radians(xfov) / 2)
    return focal_length

def parse_dae(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = {'collada': root.tag.split('}')[0].strip('{')}

    up_axis = root.find('.//collada:up_axis', ns).text
    y_up = up_axis == 'Y_UP'

    return root, ns, y_up

def get_cam_intrinsics(root, ns):
    cameras = {}
    for camera in root.findall('.//collada:camera', ns):
        camera_id = camera.get('id')
        xfov = float(camera.find('.//collada:xfov', ns).text)
        aspect_ratio = float(camera.find('.//collada:aspect_ratio', ns).text)
        cameras[camera_id] = {
            'xfov': xfov,
            'aspect_ratio': aspect_ratio
        }
    return cameras

def get_cam_extrinsics(root, ns):
    camera_nodes = []
    for node in root.findall('.//collada:node', ns):
        instance_camera = node.find('.//collada:instance_camera', ns)
        if instance_camera is not None:
            camera_id = instance_camera.get('url')[1:]  # Remove leading '#'
            matrix_text = node.find('.//collada:matrix', ns).text
            matrix = np.fromstring(matrix_text, sep=' ').reshape((4, 4))
            node_name = node.get('name')
            camera_nodes.append({
                'node_name': node_name,
                'matrix': matrix,
                'camera_id': camera_id
            })
    return camera_nodes


def create_cams_dict(cam_poses, cam_intrinsics, y_up):
    cams_dict = {}
    for i, node in enumerate(cam_poses):
        camera_id = node['camera_id']
        if camera_id in cam_intrinsics:
            camera_info = cam_intrinsics[camera_id]
            matrix = node['matrix']

            if y_up:
                matrix = convert_y_up_to_z_up(matrix)

            position = matrix[:3, 3]
            rotation_matrix = matrix[:3, :3]
            euler_angles = rotation_matrix_to_euler(rotation_matrix)

            # Convert to yaw-pitch-roll
            # TODO: check if this is correct
            heading, pitch, roll = euler_angles[1], euler_angles[0], euler_angles[2]

            # Calculate focal length from field of view
            # TODO: not sure if the value is 35mm equivalent
            focal_length = calculate_focal_length(camera_info['xfov'], 36)  

            cams_dict[i] = {
                'name': node['node_name'],
                'position': position,
                'heading': heading,
                'pitch': pitch,
                'roll': roll,
                'focal_length': focal_length
            }
    return cams_dict


def write_camera_params_to_csv(cams_dict, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['#name', 'x', 'y', 'alt', 'heading', 'pitch', 'roll', 'f', 'px', 'py', 'k1', 'k2', 'k3', 'k4', 't1', 't2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for camera in cams_dict.values():
            writer.writerow({
                '#name': camera['name'],
                'x': camera['position'][0],
                'y': camera['position'][1],
                'alt': camera['position'][2],
                'heading': round(camera['heading'], 8),
                'pitch': round(camera['pitch'], 8),
                'roll': round(camera['roll'], 8),
                'f': round(camera['focal_length'], 8),
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
    root, ns, y_up = parse_dae('assets/scene.dae')
    cam_intrinsics = get_cam_intrinsics(root, ns)
    cam_poses      = get_cam_extrinsics(root, ns)
    cams_dict      = create_cams_dict(cam_poses, cam_intrinsics, y_up)
    write_camera_params_to_csv(cams_dict, 'output/cameras_v1.csv')
