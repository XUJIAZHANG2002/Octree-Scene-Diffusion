"""SemanticKITTI ingest: raw Velodyne scans + labels -> a [256,256,32] voxel grid.

Recovered from the original SemCityOcto working tree; the outdoor notebooks in
notebooks/kitti/ import `velodene_to_voxel` from here and could not run without it.

`velodene_to_voxel` expects `semantic-kitti.yaml` at <data_path>/semantic-kitti.yaml
(a copy lives in dataset/semantic-kitti.yaml).
"""

import os
import pathlib
import sys
from typing import Tuple

import numpy as np
import open3d as o3d
import torch
import yaml


# --- Configuration ---
# Define the voxel grid parameters for SemanticKITTI scene completion task

VOXEL_SIZE = 0.2  # meters
VOXEL_GRID_DIMENSIONS = (256, 256, 32)
LIDAR_RANGE = 51.2  # meters
# GRID_ORIGIN = np.array([-LIDAR_RANGE, -LIDAR_RANGE, -2.0])
GRID_ORIGIN = np.array([0, -128*VOXEL_SIZE, 0])
SIDE_BY_SIDE_SHIFT = np.array([-128*VOXEL_SIZE, 0, 0])

def unpack_voxels(compressed):
    """
    Given a bit-encoded voxel grid, unpacks it into a normal voxel grid.
    This function is adapted from the SemKITTI devkit to read the compressed voxel format.
    """
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1
    return uncompressed

def create_voxel_grid_from_labels(labels_3d, color_map, voxel_size=1.0, z_min=-2.0):
    """
    Creates a colored Open3D VoxelGrid from a 3D NumPy array of labels.
    """
    # Find coordinates of all voxels that are NOT empty and NOT invalid (label 0 and 255)
    voxel_coords = np.argwhere((labels_3d != 255) & (labels_3d != 0))
    labels = labels_3d[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
    
    # Map labels to colors using the provided color_map
    colors = np.array([color_map.get(l, [0.0, 0.0, 0.0]) for l in labels])
    
    # Create a temporary point cloud from the voxel coordinates and colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_coords  * voxel_size)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create the VoxelGrid from the point cloud.
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=voxel_size, # Voxel size of 1.0 since coords are already integers
        min_bound=np.zeros(3),
        max_bound=np.array(VOXEL_GRID_DIMENSIONS)
    )
    
    return voxel_grid

def visualize_side_by_side_voxels(data_path, sequence, scan_index):
    """
    Loads raw LiDAR data and ground truth voxel data, generates a new voxel map,
    and visualizes them side-by-side.
    """
    print(f"Loading data for sequence {sequence}, scan {scan_index}...")
    
    # --- 1. Load Raw Lidar Data, Labels, and YAML Config ---
    raw_lidar_path = os.path.join(data_path, 'sequences', sequence, 'velodyne', f'{scan_index}.bin')
    raw_label_path = os.path.join(data_path, 'sequences', sequence, 'labels', f'{scan_index}.label')
    yaml_path = os.path.join(data_path, 'semantic-kitti.yaml')

    if not os.path.exists(raw_lidar_path) or not os.path.exists(raw_label_path):
        print("Error: Raw LiDAR or label file not found.")
        return
    if not os.path.exists(yaml_path):
        print("Error: `semantic-kitti.yaml` file not found. Please ensure it's in your dataset root.")
        return

    # Load and process the labels, using the learning map for remapping
    with open(yaml_path, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    
    remapdict = semkittiyaml['learning_map']
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    remap_lut[remap_lut == 0] = 255
    remap_lut[0] = 0

    # Load the official color map and build a remapped color map
    color_map_official = semkittiyaml['color_map']
    color_map_remapped = {v: np.array(color_map_official[k][::-1]) / 255.0 for k, v in semkittiyaml['learning_map'].items()}
    # The default color for the "unlabeled" class (0) should be defined
    color_map_remapped[0] = np.array([0,0,0]) / 255.0
    
    raw_labels = np.fromfile(raw_label_path, dtype=np.uint32)
    mapped_labels = remap_lut[raw_labels & 0xFFFF]
    points = np.fromfile(raw_lidar_path, dtype=np.float32).reshape(-1, 4)

    # --- 2. Load Ground Truth Voxel Data (for comparison) ---
    gt_voxel_label_path = os.path.join(data_path, 'dataset', 'sequences', sequence, 'voxels', f'{scan_index}.label')
    gt_invalid_path = os.path.join(data_path, 'dataset', 'sequences', sequence, 'voxels', f'{scan_index}.invalid')
    
    if not os.path.exists(gt_voxel_label_path) or not os.path.exists(gt_invalid_path):
        print("Error: Ground truth voxel data not found. Please download the 'Semantic Scene Completion' data.")
        return

    # Unpack the ground truth voxel labels and invalid voxels
    gt_voxel_labels_flat = np.fromfile(gt_voxel_label_path, dtype=np.uint16)
    gt_invalid_flat = unpack_voxels(np.fromfile(gt_invalid_path, dtype=np.uint8))
    gt_voxel_labels_flat = remap_lut[gt_voxel_labels_flat & 0xFFFF]
    gt_voxel_labels = gt_voxel_labels_flat.reshape(VOXEL_GRID_DIMENSIONS)
    gt_invalid_voxels = gt_invalid_flat.reshape(VOXEL_GRID_DIMENSIONS)
    
    # Set invalid voxels to an ignore label
    gt_voxel_labels[gt_invalid_voxels == 1] = 255

    # --- 3. Create Our Own Voxel Grid from Raw Lidar ---
    # Convert point cloud to a VoxelGrid using the same parameters as the GT.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd_colors = np.array([color_map_remapped.get(l, [0, 0, 0]) for l in mapped_labels])
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    # Apply the translation to the point cloud before creating the voxel grid
    pcd.translate(SIDE_BY_SIDE_SHIFT)

    # Use create_from_point_cloud_within_bounds to explicitly set the boundaries
    my_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=VOXEL_SIZE,
        min_bound=GRID_ORIGIN + SIDE_BY_SIDE_SHIFT,
        max_bound=GRID_ORIGIN + np.array(VOXEL_GRID_DIMENSIONS) * VOXEL_SIZE + SIDE_BY_SIDE_SHIFT
    )

    gt_voxel_labels[gt_voxel_labels == 255] = 0
    print(np.unique(gt_voxel_labels))
    # --- 4. Create GT Voxel Grid from Ground Truth Labels ---
    gt_voxel_grid = create_voxel_grid_from_labels(gt_voxel_labels, color_map_remapped)

    # --- 5. Visualize Both Voxel Grids ---
    print("\nVisualizing both voxel maps side-by-side.")
    print("Left: Voxel map generated from raw LiDAR scan.")
    print("Right: Ground truth voxel map (from Semantic Scene Completion data).")
    
    o3d.visualization.draw_geometries([my_voxel_grid, gt_voxel_grid])
    
    
def visualize_aligned_voxels(data_path, sequence, scan_index):
    """
    Loads raw LiDAR data and ground truth voxel data, generates a new voxel map,
    and visualizes them side-by-side.
    """
    print(f"Loading data for sequence {sequence}, scan {scan_index}...")
    
    # --- 1. Load Raw Lidar Data, Labels, and YAML Config ---
    raw_lidar_path = os.path.join(data_path, 'sequences', sequence, 'velodyne', f'{scan_index}.bin')
    raw_label_path = os.path.join(data_path, 'sequences', sequence, 'labels', f'{scan_index}.label')
    yaml_path = os.path.join(data_path, 'semantic-kitti.yaml')

    if not os.path.exists(raw_lidar_path) or not os.path.exists(raw_label_path):
        print("Error: Raw LiDAR or label file not found.")
        return
    if not os.path.exists(yaml_path):
        print("Error: `semantic-kitti.yaml` file not found. Please ensure it's in your dataset root.")
        return

    # Load and process the labels, using the learning map for remapping
    with open(yaml_path, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    
    remapdict = semkittiyaml['learning_map']
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    remap_lut[remap_lut == 0] = 255
    remap_lut[0] = 0

    # Load the official color map and build a remapped color map
    color_map_official = semkittiyaml['color_map']
    color_map_remapped = {v: np.array(color_map_official[k][::-1]) / 255.0 for k, v in semkittiyaml['learning_map'].items()}
    # The default color for the "unlabeled" class (0) should be defined
    color_map_remapped[0] = np.array([0,0,0]) / 255.0
    
    raw_labels = np.fromfile(raw_label_path, dtype=np.uint32)
    mapped_labels = remap_lut[raw_labels & 0xFFFF]
    points = np.fromfile(raw_lidar_path, dtype=np.float32).reshape(-1, 4)

    # --- 2. Load Ground Truth Voxel Data (for comparison) ---
    gt_voxel_label_path = os.path.join(data_path, 'dataset', 'sequences', sequence, 'voxels', f'{scan_index}.label')
    gt_invalid_path = os.path.join(data_path, 'dataset', 'sequences', sequence, 'voxels', f'{scan_index}.invalid')
    
    if not os.path.exists(gt_voxel_label_path) or not os.path.exists(gt_invalid_path):
        print("Error: Ground truth voxel data not found. Please download the 'Semantic Scene Completion' data.")
        return

    # Unpack the ground truth voxel labels and invalid voxels
    gt_voxel_labels_flat = np.fromfile(gt_voxel_label_path, dtype=np.uint16)
    gt_invalid_flat = unpack_voxels(np.fromfile(gt_invalid_path, dtype=np.uint8))
    gt_voxel_labels_flat = remap_lut[gt_voxel_labels_flat & 0xFFFF]
    gt_voxel_labels = gt_voxel_labels_flat.reshape(VOXEL_GRID_DIMENSIONS)
    gt_invalid_voxels = gt_invalid_flat.reshape(VOXEL_GRID_DIMENSIONS)
    
    # Set invalid voxels to an ignore label
    gt_voxel_labels[gt_invalid_voxels == 1] = 255

    # --- 3. Create Our Own Voxel Grid from Raw Lidar ---
    # Convert point cloud to a VoxelGrid using the same parameters as the GT.

    mapped_labels = mapped_labels[points[:, 0] > 0]
    points = points[points[:, 0] > 0]
    mapped_labels = mapped_labels[points[:, 1] < 128 * VOXEL_SIZE]
    points = points[points[:, 1] < 128 * VOXEL_SIZE]
    mapped_labels = mapped_labels[points[:, 1] > -128 * VOXEL_SIZE]
    points = points[points[:, 1] > -128 * VOXEL_SIZE]
    mapped_labels = mapped_labels[points[:, 0] < 256 * VOXEL_SIZE]
    points = points[points[:, 0] < 256 * VOXEL_SIZE]
    
    z_min = min(map(lambda x: x[2], points[np.where(mapped_labels == 9)])) #np.argwhere(points == 40)
    mapped_labels = mapped_labels[points[:, 2] > z_min]
    points = points[points[:, 2] > z_min]
    
    mapped_labels = mapped_labels[points[:, 2] < z_min + 32 * VOXEL_SIZE]
    points = points[points[:, 2] < z_min + 32 * VOXEL_SIZE]
    
    
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd_colors = np.array([color_map_remapped.get(l, [0, 0, 0]) for l in mapped_labels])
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    # Apply the translation to the point cloud before creating the voxel grid
    shift = np.array([0, -128 * VOXEL_SIZE, -z_min])
    pcd.translate(shift)

    # Use create_from_point_cloud_within_bounds to explicitly set the boundaries
    my_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=0.2,
        min_bound=np.zeros(3),
        max_bound=np.array(VOXEL_GRID_DIMENSIONS)
        # min_bound=GRID_ORIGIN + SIDE_BY_SIDE_SHIFT,
        # max_bound=GRID_ORIGIN + np.array(VOXEL_GRID_DIMENSIONS) * VOXEL_SIZE + SIDE_BY_SIDE_SHIFT
    )

    gt_voxel_labels[gt_voxel_labels == 255] = 0
    print(np.unique(gt_voxel_labels))
    # --- 4. Create GT Voxel Grid from Ground Truth Labels ---
    gt_voxel_grid = create_voxel_grid_from_labels(gt_voxel_labels, color_map_remapped, voxel_size=VOXEL_SIZE)

    # --- 5. Visualize Both Voxel Grids ---
    print("\nVisualizing both voxel maps side-by-side.")
    print("Left: Voxel map generated from raw LiDAR scan.")
    print("Right: Ground truth voxel map (from Semantic Scene Completion data).")
    
    o3d.visualization.draw_geometries([my_voxel_grid, gt_voxel_grid])
def create_voxel_grid_from_labels(labels_3d, color_map, min_bound=None, max_bound=None, voxel_size=None):
    """
    Creates a colored Open3D VoxelGrid from a 3D NumPy array of labels.
    """
    # Find coordinates of all voxels that are NOT empty and NOT invalid (label 0 and 255)
    voxel_coords = np.argwhere((labels_3d != 255) & (labels_3d != 0))
    labels = labels_3d[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
    
    # Map labels to colors using the provided color_map
    colors = np.array([color_map.get(l, [0.0, 0.0, 0.0]) for l in labels])
    
    # Create a temporary point cloud from the voxel coordinates and colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_coords * voxel_size)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create the VoxelGrid from the point cloud within the given bounds.
    # The voxel coordinates are in an index-based grid, so the voxel_size is 1.0.
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=voxel_size, 
        min_bound=np.zeros(3),
        max_bound=np.array(VOXEL_GRID_DIMENSIONS)
    )
    
    return voxel_grid

def get_voxel_maps(data_path, sequence, scan_index):
    """
    Loads raw and ground truth data and creates both the raw point-based voxel map
    and the ground truth voxel map.
    """
    raw_lidar_path = os.path.join(data_path, 'sequences', sequence, 'velodyne', f'{scan_index}.bin')
    raw_label_path = os.path.join(data_path, 'sequences', sequence, 'labels', f'{scan_index}.label')
    yaml_path = os.path.join(data_path, 'semantic-kitti.yaml')

    if not all(os.path.exists(p) for p in [raw_lidar_path, raw_label_path, yaml_path]):
        print("Error: Missing dataset files. Please check your data_path and file structure.")
        sys.exit(1)
        
    with open(yaml_path, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    
    remapdict = semkittiyaml['learning_map']
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    remap_lut[remap_lut == 0] = 255
    remap_lut[0] = 0

    color_map_official = semkittiyaml['color_map']
    color_map_remapped = {v: np.array(color_map_official[k][::-1]) / 255.0 for k, v in semkittiyaml['learning_map'].items()}
    color_map_remapped[0] = np.array([0,0,0]) / 255.0

    raw_labels = np.fromfile(raw_label_path, dtype=np.uint32)
    mapped_labels = remap_lut[raw_labels & 0xFFFF]
    points = np.fromfile(raw_lidar_path, dtype=np.float32).reshape(-1, 4)

    # # Filter points to be within the GT voxel grid bounds
    # points_in_bounds = (points[:, 0] >= GRID_ORIGIN[0]) & (points[:, 0] < GRID_ORIGIN[0] + VOXEL_GRID_DIMENSIONS[0] * VOXEL_SIZE) & \
    #                    (points[:, 1] >= GRID_ORIGIN[1]) & (points[:, 1] < GRID_ORIGIN[1] + VOXEL_GRID_DIMENSIONS[1] * VOXEL_SIZE) & \
    #                    (points[:, 2] >= GRID_ORIGIN[2]) & (points[:, 2] < GRID_ORIGIN[2] + VOXEL_GRID_DIMENSIONS[2] * VOXEL_SIZE)
                       
    # filtered_points = points[points_in_bounds, :3]
    # filtered_labels = mapped_labels[points_in_bounds]
    
    
    mapped_labels = mapped_labels[points[:, 0] > 0]
    points = points[points[:, 0] > 0]
    mapped_labels = mapped_labels[points[:, 1] < 128 * VOXEL_SIZE]
    points = points[points[:, 1] < 128 * VOXEL_SIZE]
    mapped_labels = mapped_labels[points[:, 1] > -128 * VOXEL_SIZE]
    points = points[points[:, 1] > -128 * VOXEL_SIZE]
    mapped_labels = mapped_labels[points[:, 0] < 256 * VOXEL_SIZE]
    points = points[points[:, 0] < 256 * VOXEL_SIZE]
    
    z_min = min(map(lambda x: x[2], points[np.where(mapped_labels == 9)])) #np.argwhere(points == 40)
    z_min = -2
    mapped_labels = mapped_labels[points[:, 2] > z_min]
    points = points[points[:, 2] > z_min]
    
    mapped_labels = mapped_labels[points[:, 2] < z_min + 32 * VOXEL_SIZE]
    points = points[points[:, 2] < z_min + 32 * VOXEL_SIZE]
    
    filtered_points = points[:, :3]
    filtered_labels = mapped_labels
    
    # Create the generated voxel map from the filtered point cloud
    
    my_pcd = o3d.geometry.PointCloud()
    my_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    my_pcd_colors = np.array([color_map_remapped.get(l, [0, 0, 0]) for l in filtered_labels])
    my_pcd.colors = o3d.utility.Vector3dVector(my_pcd_colors)
    shift = np.array([0, 128 * VOXEL_SIZE, 2])
    my_pcd.translate(shift)
    print(np.min(my_pcd.points, axis=0))
    print(np.max(my_pcd.points, axis=0))
    my_voxel_grid_3d = np.zeros(VOXEL_GRID_DIMENSIONS, dtype=np.uint8)
    for i, p in enumerate(my_pcd.points):
        x, y, z = p / VOXEL_SIZE
        ix, iy, iz = int(x), int(y), int(z)
        if 0 <= ix < 256 and 0 <= iy < 256 and 0 <= iz < 32:
            my_voxel_grid_3d[ix, iy, iz] = filtered_labels[i]

    # --- Load Ground Truth Voxel Data ---
    gt_voxel_label_path = os.path.join(data_path, 'dataset', 'sequences', sequence, 'voxels', f'{scan_index}.label')
    gt_invalid_path = os.path.join(data_path, 'dataset', 'sequences', sequence, 'voxels', f'{scan_index}.invalid')
    
    gt_voxel_labels_flat = np.fromfile(gt_voxel_label_path, dtype=np.uint16)
    gt_invalid_flat = unpack_voxels(np.fromfile(gt_invalid_path, dtype=np.uint8))
    gt_voxel_labels_flat = remap_lut[gt_voxel_labels_flat & 0xFFFF]
    gt_voxel_labels = gt_voxel_labels_flat.reshape(VOXEL_GRID_DIMENSIONS)
    gt_invalid_voxels = gt_invalid_flat.reshape(VOXEL_GRID_DIMENSIONS)
    
    gt_voxel_labels[gt_invalid_voxels == 1] = 255
    gt_voxel_labels_flat[gt_voxel_labels_flat == 255] = 0
    
    return my_voxel_grid_3d, gt_voxel_labels, color_map_remapped
    
def visualize_all_maps(my_voxel_labels, gt_voxel_labels, color_map_remapped):
    """
    Visualizes the raw point cloud, the GT voxel map, and the differences.
    """
    # Create Open3D voxel grids from the label arrays
    my_voxel_grid = create_voxel_grid_from_labels(my_voxel_labels, color_map_remapped, GRID_ORIGIN, GRID_ORIGIN + np.array(VOXEL_GRID_DIMENSIONS) * VOXEL_SIZE, VOXEL_SIZE)
    gt_voxel_grid = create_voxel_grid_from_labels(gt_voxel_labels, color_map_remapped, GRID_ORIGIN, GRID_ORIGIN + np.array(VOXEL_GRID_DIMENSIONS) * VOXEL_SIZE, VOXEL_SIZE)

    # Shift for side-by-side visualization
    my_voxel_grid.translate(np.array([-10, 0, 0]))
    gt_voxel_grid.translate(np.array([10, 0, 0]))

    print("\nVisualizing both voxel maps side-by-side.")
    print("Left: Voxel map generated from raw LiDAR scan.")
    print("Right: Ground truth voxel map (from Semantic Scene Completion data).")
    
    o3d.visualization.draw_geometries([my_voxel_grid, gt_voxel_grid])
    
def velodene_to_voxel(
    data_path: str,
    sequence: str,
    scan_index: str,
    voxel_size: float = VOXEL_SIZE,
    voxel_grid_dimensions: Tuple[int, int, int] = VOXEL_GRID_DIMENSIONS):
    """
    Loads raw and ground truth data and creates both the raw point-based voxel map
    and the ground truth voxel map.
    """
    raw_lidar_path = os.path.join(data_path, 'sequences', sequence, 'velodyne', f'{scan_index}.bin')
    raw_label_path = os.path.join(data_path, 'sequences', sequence, 'labels', f'{scan_index}.label')
    yaml_path = os.path.join(data_path, 'semantic-kitti.yaml')
    print(f"Loading data for sequence {sequence}, scan {scan_index}...")
    

    if not all(os.path.exists(p) for p in [raw_lidar_path, raw_label_path, yaml_path]):
        print("Error: Missing dataset files. Please check your data_path and file structure.")
        
    with open(yaml_path, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    
    remapdict = semkittiyaml['learning_map']
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    remap_lut[remap_lut == 0] = 255
    remap_lut[0] = 0

    color_map_official = semkittiyaml['color_map']
    color_map_remapped = {v: np.array(color_map_official[k][::-1]) / 255.0 for k, v in semkittiyaml['learning_map'].items()}
    color_map_remapped[0] = np.array([0,0,0]) / 255.0

    raw_labels = np.fromfile(raw_label_path, dtype=np.uint32)
    mapped_labels = remap_lut[raw_labels & 0xFFFF]
    points = np.fromfile(raw_lidar_path, dtype=np.float32).reshape(-1, 4)
    
    
    mapped_labels = mapped_labels[points[:, 0] > 0]
    points = points[points[:, 0] > 0]
    mapped_labels = mapped_labels[points[:, 1] < 128 * VOXEL_SIZE]
    points = points[points[:, 1] < 128 * VOXEL_SIZE]
    mapped_labels = mapped_labels[points[:, 1] > -128 * VOXEL_SIZE]
    points = points[points[:, 1] > -128 * VOXEL_SIZE]
    mapped_labels = mapped_labels[points[:, 0] < 256 * VOXEL_SIZE]
    points = points[points[:, 0] < 256 * VOXEL_SIZE]
    z_min = -2
    mapped_labels = mapped_labels[points[:, 2] > z_min]
    points = points[points[:, 2] > z_min]
    
    mapped_labels = mapped_labels[points[:, 2] < z_min + 32 * VOXEL_SIZE]
    points = points[points[:, 2] < z_min + 32 * VOXEL_SIZE]
    
    filtered_points = points[:, :3]
    filtered_labels = mapped_labels
    
    # Create the generated voxel map from the filtered point cloud
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd_colors = np.array([color_map_remapped.get(l, [0, 0, 0]) for l in filtered_labels])
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    shift = np.array([0, 128 * voxel_size, 2])
    pcd.translate(shift)
    voxel_grid_3d = np.zeros(voxel_grid_dimensions, dtype=np.uint8)
    for i, p in enumerate(pcd.points):
        x, y, z = p / voxel_size
        ix, iy, iz = int(x), int(y), int(z)
        if 0 <= ix < 256 and 0 <= iy < 256 and 0 <= iz < 32:
            voxel_grid_3d[ix, iy, iz] = filtered_labels[i]
    
    return voxel_grid_3d
    


def visualize_difference_voxels(data_path, sequence, scan_index):
    """
    Generates and visualizes a new voxel map showing only the voxels that are
    in the ground truth but not in the raw LiDAR scan.
    """
    print(f"Loading data for sequence {sequence}, scan {scan_index}...")

    my_voxel_labels, gt_voxel_labels, color_map_remapped = get_voxel_maps(data_path, sequence, scan_index)
    print(my_voxel_labels.shape, gt_voxel_labels.shape)
    print(np.unique(my_voxel_labels), np.unique(gt_voxel_labels))
    # Create a new label array for the difference map
    difference_labels = np.zeros(VOXEL_GRID_DIMENSIONS, dtype=np.uint8)

    # Find where the GT is occupied (and not invalid) but our map is empty
    # difference_mask = (gt_voxel_labels != 0) & (gt_voxel_labels != 255) & (my_voxel_labels == 0)
    my_voxel_labels[my_voxel_labels == 255] = 0
    union_mask = (gt_voxel_labels != 0) & (my_voxel_labels != 0)
    print(union_mask.sum(), (my_voxel_labels != 0).sum(), (gt_voxel_labels != 0).sum())
    difference_mask = (gt_voxel_labels != 0) & (my_voxel_labels == 0)
    
    # Assign the GT labels to the difference map where the mask is true
    # difference_labels[difference_mask] = gt_voxel_labels[difference_mask]
    difference_labels[union_mask] = gt_voxel_labels[union_mask]
    
    # Now, create a new Open3D voxel grid from this difference map
    difference_voxel_grid = create_voxel_grid_from_labels(difference_labels, color_map_remapped, GRID_ORIGIN, GRID_ORIGIN + np.array(VOXEL_GRID_DIMENSIONS) * VOXEL_SIZE, VOXEL_SIZE)
    
    print("\nVisualizing only the voxels that are in the GT but not in the scan.")
    print("This reveals occluded or missing parts of the scene.")
    o3d.visualization.draw_geometries([difference_voxel_grid])


def visualize_instance(voxel_map):
    voxel_map[voxel_map > 0] =  20
    from octree_diff.viz.open3d_viewers import visualize_kitti_instance
    visualize_kitti_instance(torch.tensor(voxel_map).cuda())

if __name__ == "__main__":
    # --- User-Defined Parameters ---
    # Change these to match your dataset location and the specific scan you want to visualize
    KITTI_DATA_PATH = "./dataset/"
    SEQUENCE_NUMBER = "08"
    SCAN_INDEX = "000000"

    # visualize_aligned_voxels(KITTI_DATA_PATH, SEQUENCE_NUMBER, SCAN_INDEX)
    # visualize_difference_voxels(KITTI_DATA_PATH, SEQUENCE_NUMBER, SCAN_INDEX)
    vox = velodene_to_voxel(KITTI_DATA_PATH, SEQUENCE_NUMBER, SCAN_INDEX)
    visualize_instance(vox)