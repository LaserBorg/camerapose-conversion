import json
import numpy as np
import os


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_ply_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    header_ended = False
    vertex_data = []
    for line in lines:
        if header_ended:
            vertex_data.append(line.strip().split())
        elif line.strip() == "end_header":
            header_ended = True
    
    return vertex_data

def rotation_matrix_to_quaternion(matrix):
    # Convert the matrix values to floats
    m = np.array(matrix, dtype=float).reshape(3, 3)
    q = np.empty((4, ))
    t = np.trace(m)
    if (t > 0):
        s = np.sqrt(t + 1.0) * 2
        q[3] = 0.25 * s
        q[0] = (m[2, 1] - m[1, 2]) / s
        q[1] = (m[0, 2] - m[2, 0]) / s
        q[2] = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q[3] = (m[2, 1] - m[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (m[0, 1] + m[1, 0]) / s
        q[2] = (m[0, 2] + m[2, 0]) / s
    elif (m[1, 1] > m[2, 2]):
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q[3] = (m[0, 2] - m[2, 0]) / s
        q[0] = (m[0, 1] + m[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q[3] = (m[1, 0] - m[0, 1]) / s
        q[0] = (m[0, 2] + m[2, 0]) / s
        q[1] = (m[1, 2] + m[2, 1]) / s
        q[2] = 0.25 * s
    return q


def create_cameras_txt(json_data, output_file):
    with open(output_file, 'w') as file:
        # Write header information
        file.write("# Camera list with one line of data per camera:\n")
        file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        file.write(f"# Number of cameras: {len(json_data['intrinsics'])}\n")
        
        # Iterate through the intrinsics list
        for i, intrinsic in enumerate(json_data['intrinsics']):
            camera_id = i + 1
            model = 'SIMPLE_RADIAL'  # intrinsic['type'].upper()
            width = int(intrinsic['width'])
            height = int(intrinsic['height'])
            focal_length_mm = float(intrinsic['focalLength'])
            sensor_width_mm = float(intrinsic['sensorWidth'])
            sensor_height_mm = float(intrinsic['sensorHeight'])
            
            # Convert focal length from mm to pixels
            focal_length_pixels = (focal_length_mm / sensor_width_mm) * width
            
            principal_point_x = width / 2 + float(intrinsic['principalPoint'][0])
            principal_point_y = height / 2 + float(intrinsic['principalPoint'][1])
            distortion_param = float(intrinsic['distortionParams'][0])
            
            # Write the camera data
            file.write(f"{camera_id} {model} {width} {height} {focal_length_pixels} {principal_point_x} {principal_point_y} {distortion_param}\n")

def create_images_txt(json_data, output_file):
    # Create a dictionary to map poseId to image path
    pose_to_path = {view['poseId']: view['path'] for view in json_data['views']}
    
    with open(output_file, 'w') as file:
        # Write header information
        file.write("# Image list with two lines of data per image:\n")
        file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        file.write(f"# Number of images: {len(json_data['poses'])}, mean observations per image: 0.0\n")  # Placeholder for mean observations
        
        # Iterate through the poses list
        for i, pose in enumerate(json_data['poses']):
            image_id = i + 1
            rotation_matrix = pose['pose']['transform']['rotation']
            quaternion = rotation_matrix_to_quaternion(rotation_matrix)
            tx, ty, tz = map(float, pose['pose']['transform']['center'])
            camera_id = 1  # Placeholder for camera ID
            name = os.path.basename(pose_to_path[pose['poseId']])  # Get the image file name
            
            # Write the image data
            file.write(f"{image_id} {quaternion[3]} {quaternion[0]} {quaternion[1]} {quaternion[2]} {tx} {ty} {tz} {camera_id} {name}\n")
            file.write("0.0 0.0 -1\n")  # Placeholder for POINTS2D

def create_points3D_txt(ply_data, output_file):
    with open(output_file, 'w') as file:
        # Write header information
        file.write("# 3D point list with one line of data per point:\n")
        file.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        file.write(f"# Number of points: {len(ply_data)}, mean track length: 0.0\n")  # Placeholder for mean track length
        
        # Iterate through the PLY data
        for i, vertex in enumerate(ply_data):
            point3d_id = i + 1
            x, y, z = vertex[0], vertex[1], vertex[2]
            r, g, b = vertex[3], vertex[4], vertex[5]
            error = 0.0  # Placeholder for error
            track = ""  # Placeholder for track
            
            # Write the point data
            file.write(f"{point3d_id} {x} {y} {z} {r} {g} {b} {error} {track}\n")


json_data = read_json_file('assets/sfm/2/cameras.sfm')  # SfM (json)
ply_data = read_ply_file('assets/sfm/2/cloud_and_poses.ply') 

create_cameras_txt(json_data, 'output/sfm/2/cameras.txt') 
create_images_txt(json_data, 'output/sfm/2/images.txt')   
create_points3D_txt(ply_data, 'output/sfm/2/points3D.txt') 
