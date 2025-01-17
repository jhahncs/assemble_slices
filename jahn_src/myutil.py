import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import open3d as o3d 
import math
import torch

near_zero_value = 8.034209978553344e-10

def get_plane(_pcs):
    min_xyz = np.min(_pcs, axis=0)
    max_xyz = np.max(_pcs, axis=0)
    
    #print(min_xyz)
    #print(max_xyz)
    points_min_z = []
    points_max_z = []
    for xyz in _pcs:
        if xyz[2] == min_xyz[2]:
            points_min_z.append(np.array(xyz))
        if xyz[2] == max_xyz[2]:
            points_max_z.append(np.array(xyz))
            
    return points_min_z[0], points_min_z[1], points_max_z[0]
    
#get_plane(pcs[0])


def calculate_normal_vector(p1, p2, p3):
    """Calculates the normal vector of a plane defined by three points.

    Args:
        p1 (np.array): The first point on the plane.
        p2 (np.array): The second point on the plane.
        p3 (np.array): The third point on the plane.

    Returns:
        np.array: The normal vector of the plane.
    """

    v1 = p2 - p1
    v2 = p3 - p1
    normal_vector = np.cross(v1, v2)
    return normal_vector
    
def calculate_normal_vector_open3d(_pcs, dist_threshold=0.02, visualize=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_pcs)
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model    
    #plane_model = [0 for plane_model if a < 0.00005 ]
    return [a,b,c]
def recenter_pc(pc):
    """pc: [N, 3]"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid[None]
    return pc, centroid

def rotate_pc(pc):
    """pc: [N, 3]"""
    rot_mat = R.random().as_matrix()
    #print('rot_mat')
    #print(rot_mat)
    pc = (rot_mat @ pc.T).T
    #print('pc')
    #print(pc.shape)
    quat_gt = R.from_matrix(rot_mat.T).as_quat()
    #print('quat_gt')
    #print(quat_gt)
    # we use scalar-first quaternion
    quat_gt = quat_gt[[3, 0, 1, 2]]
    #print(quat_gt)
    return pc, quat_gt

def rotate_and_translate(pc):
    pc, _ = recenter_pc(pc)
    pc, _ = rotate_pc(pc)
    return pc
def get_noise_trans_rots_concat(trans_rot_shape, device):
    noise_trans_rots = torch.randn(trans_rot_shape, device=device) # B(insance), N(max parts), trans+rot
    
        #noise_trans = torch.randn((2, 4))
    noise_trans_rots[:,:,4:6] = 0.0
    noise_trans_rots[:,:,1] = 0.0

    return noise_trans_rots

def get_noise_trans_rots(trans_shape, rot_shape):
    noise_trans = torch.randn(trans_shape)
    noise_rots = torch.randn(rot_shape)
        #noise_trans = torch.randn((2, 4))
    noise_rots[:,1:3] = 0.0
    noise_trans[:,1] = 0.0

    return noise_trans, noise_rots


def rotate_and_translate_to_xy_plane(points, normal_vector, translation_point=None, center_point = None, debug=False):
    
    # Ensure the normal vector is a unit vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    normal_vector = np.array([0.0 if a < near_zero_value else a for a in normal_vector])
    if debug: 
        print('normal_vector',normal_vector)
    # Define the target normal vector (Z-axis)
    #target_vector = np.array([0.3, 1, 0.5])
    
    # Calculate the rotation axis using cross product
    #rotation_axis = np.cross(normal_vector, target_vector)
    #if debug: 
    #    print('rotation_axis',rotation_axis)
    rotation_axis = normal_vector
    #rotation_axis = np.array([0.0 if a < 8.034209978553344e-10 else a for a in rotation_axis])

    # Calculate the angle between the normal vector and the target vector
    #angle = np.arccos(np.clip(np.dot(normal_vector, target_vector), -1.0, 1.0))
    angle = random.random()*math.pi
    
    if debug: 
        print('rotation_axis',rotation_axis)
        print('angle',angle)
        print('np.linalg.norm(rotation_axis)',np.linalg.norm(rotation_axis))
        print(rotation_axis * angle)
    
    # Create the rotation using the axis and angle
    if np.linalg.norm(rotation_axis) != 0:
        rotation = R.from_rotvec(rotation_axis * angle)
    else:
        rotation = R.from_rotvec([0, 0, 0])
    if debug: 
        print('rotation',rotation.as_rotvec())
    # Apply the rotation to the points
    rotated_points = np.array(rotation.apply(points))
    
    # If no specific translation point is provided, use the centroid
    #if translation_point is None:
    #    translation_point = np.mean(rotated_points, axis=0)

    

    quat_gt = rotation.as_quat()
        # we use scalar-first quaternion
    quat_gt = quat_gt[[3, 0, 1, 2]]


    
    # Translate points so that the specified point lies on the XY plane
    #translation_vector = np.array([0, 0, -translation_point[2]])
    #translation_vector = np.array([random.random(), random.random(), -translation_point[2]])
    
    if center_point is None:
        #translation_vector = np.array([random.random(), random.random(), random.random() ])
        translation_vector = np.array([random.random(), 0, random.random() ])
    else:
        #translation_vector = np.array([center_point[0], -translation_point[1], center_point[2] ])
        translation_vector = np.array([center_point[0], 0, center_point[2] ])
    #rotated_points = points
    #print(rotated_points.shape)
    #print(translation_vector.shape)
    translated_points = rotated_points + translation_vector
    #print(translated_points.shape)
    #translated_points = rotated_points
    #translated_points, _ = recenter_pc(translated_points)
    if debug: 
        print('translation_vector',translation_vector)
        print('quat_gt',quat_gt)
    return translated_points, translation_vector, quat_gt
